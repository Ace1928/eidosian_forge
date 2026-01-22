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
class FrozenEstimator(BaseEstimator):

    def __init__(self, fitted_estimator):
        self.fitted_estimator = fitted_estimator

    def __getattr__(self, name):
        return getattr(self.fitted_estimator, name)

    def __sklearn_clone__(self):
        return self

    def fit(self, *args, **kwargs):
        return self

    def fit_transform(self, *args, **kwargs):
        return self.fitted_estimator.transform(*args, **kwargs)