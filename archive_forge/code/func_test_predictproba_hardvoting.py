import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_multilabel_classification
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
def test_predictproba_hardvoting():
    eclf = VotingClassifier(estimators=[('lr1', LogisticRegression()), ('lr2', LogisticRegression())], voting='hard')
    inner_msg = "predict_proba is not available when voting='hard'"
    outer_msg = "'VotingClassifier' has no attribute 'predict_proba'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        eclf.predict_proba
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)
    assert not hasattr(eclf, 'predict_proba')
    eclf.fit(X_scaled, y)
    assert not hasattr(eclf, 'predict_proba')