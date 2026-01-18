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
def test_set_params_passes_all_parameters():

    class TestDecisionTree(DecisionTreeClassifier):

        def set_params(self, **kwargs):
            super().set_params(**kwargs)
            assert kwargs == expected_kwargs
            return self
    expected_kwargs = {'max_depth': 5, 'min_samples_leaf': 2}
    for est in [Pipeline([('estimator', TestDecisionTree())]), GridSearchCV(TestDecisionTree(), {})]:
        est.set_params(estimator__max_depth=5, estimator__min_samples_leaf=2)