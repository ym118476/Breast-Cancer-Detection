import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  , classification_report
import seaborn as sns

# Load dataset
data = load_breast_cancer()

features = data.data
labels = data.target
label_names = data["target_names"]
feature_names = data["feature_names"]

sns.heatmap(pd.DataFrame(features,columns=feature_names))

# Standardize features
standard = StandardScaler()
features = standard.fit_transform(features)  # Apply scaling directly to features




print(features.shape)  #
print(labels.shape)    

# Split data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, random_state=42)

# Fit a Logistic Regression model (uncomment this if you want to run it)
logisticRegression = LogisticRegression(max_iter=100)
logisticRegression.fit(xtrain, ytrain)
pred_Lr=logisticRegression.predict(xtest)
print(logisticRegression.score(xtest, ytest))

# support vector clssifier
svc=SVC()
svc.fit(xtrain, ytrain)
pred_svc=svc.predict(xtest)
print(svc.score(xtest, ytest))

#random forest classifier 
randomforest=RandomForestClassifier(n_estimators=200)
randomforest.fit(xtrain, ytrain)
pred_rf=randomforest.predict(xtest)
print(randomforest.score(xtest, ytest))

#decision tree classifier 
decisiontree=DecisionTreeClassifier()
decisiontree.fit(xtrain, ytrain)
pred_decision_tree=decisiontree.predict(xtest)
print(decisiontree.score(xtest, ytest))


cm1=confusion_matrix(ytest,pred_Lr)
sns.heatmap(cm1)

f1_Score=f1_score(ytest, pred_Lr)
print(f1_Score)

precision=precision_score(ytest, pred_Lr)
recall=recall_score(ytest, pred_Lr)

Classification_report=classification_report(ytest,pred_Lr)

print(f"classification_report is {Classification_report}")