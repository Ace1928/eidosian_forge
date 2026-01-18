import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.parallel import Parallel
def test_sparse_coder_n_features_in():
    d = np.array([[1, 2, 3], [1, 2, 3]])
    sc = SparseCoder(d)
    assert sc.n_features_in_ == d.shape[1]