from math import ceil
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris, make_blobs
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
Check that we raise the proper AttributeErrors when the `base_estimator`
    does not implement the `predict_proba` method, which is called from within
    `fit`, or `decision_function`, which is decorated with `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    