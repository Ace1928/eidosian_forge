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
def test_pickling_when_getstate_is_overwritten_by_mixin_outside_of_sklearn():
    try:
        estimator = MultiInheritanceEstimator()
        text = 'this attribute should not be pickled'
        estimator._attribute_not_pickled = text
        old_mod = type(estimator).__module__
        type(estimator).__module__ = 'notsklearn'
        serialized = estimator.__getstate__()
        assert serialized == {'_attribute_not_pickled': None, 'attribute_pickled': 5}
        serialized['attribute_pickled'] = 4
        estimator.__setstate__(serialized)
        assert estimator.attribute_pickled == 4
        assert estimator._restored
    finally:
        type(estimator).__module__ = old_mod