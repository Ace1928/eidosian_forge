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
def test_pickle_version_warning_is_issued_when_no_version_info_in_pickle():
    iris = datasets.load_iris()
    tree = TreeNoVersion().fit(iris.data, iris.target)
    tree_pickle_noversion = pickle.dumps(tree)
    assert b'_sklearn_version' not in tree_pickle_noversion
    message = pickle_error_message.format(estimator='TreeNoVersion', old_version='pre-0.18', current_version=sklearn.__version__)
    with pytest.warns(UserWarning, match=message):
        pickle.loads(tree_pickle_noversion)