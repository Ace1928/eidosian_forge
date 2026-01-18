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
def test_pickle_version_warning_is_issued_upon_different_version():
    iris = datasets.load_iris()
    tree = TreeBadVersion().fit(iris.data, iris.target)
    tree_pickle_other = pickle.dumps(tree)
    message = pickle_error_message.format(estimator='TreeBadVersion', old_version='something', current_version=sklearn.__version__)
    with pytest.warns(UserWarning, match=message) as warning_record:
        pickle.loads(tree_pickle_other)
    message = warning_record.list[0].message
    assert isinstance(message, InconsistentVersionWarning)
    assert message.estimator_name == 'TreeBadVersion'
    assert message.original_sklearn_version == 'something'
    assert message.current_sklearn_version == sklearn.__version__