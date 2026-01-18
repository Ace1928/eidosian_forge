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
def test_pickling_when_getstate_is_overwritten_by_mixin():
    estimator = MultiInheritanceEstimator()
    estimator._attribute_not_pickled = 'this attribute should not be pickled'
    serialized = pickle.dumps(estimator)
    estimator_restored = pickle.loads(serialized)
    assert estimator_restored.attribute_pickled == 5
    assert estimator_restored._attribute_not_pickled is None
    assert estimator_restored._restored