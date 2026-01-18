import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
def test_raises_on_get_params_non_attribute():

    class MyEstimator(BaseEstimator):

        def __init__(self, param=5):
            pass

        def fit(self, X, y=None):
            return self
    est = MyEstimator()
    msg = "'MyEstimator' object has no attribute 'param'"
    with pytest.raises(AttributeError, match=msg):
        est.get_params()