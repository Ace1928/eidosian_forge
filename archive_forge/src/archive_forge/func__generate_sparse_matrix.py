import pickle
import re
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from .. import config_context
from ..base import (
from ..datasets import (
from ..exceptions import DataConversionWarning, NotFittedError, SkipTestWarning
from ..feature_selection import SelectFromModel, SelectKBest
from ..linear_model import (
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from ..model_selection import ShuffleSplit, train_test_split
from ..model_selection._validation import _safe_split
from ..pipeline import make_pipeline
from ..preprocessing import StandardScaler, scale
from ..random_projection import BaseRandomProjection
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils._array_api import (
from ..utils._array_api import (
from ..utils._param_validation import (
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import check_is_fitted
from . import IS_PYPY, is_scalar_nan, shuffle
from ._param_validation import Interval
from ._tags import (
from ._testing import (
from .validation import _num_samples, has_fit_parameter
def _generate_sparse_matrix(X_csr):
    """Generate sparse matrices with {32,64}bit indices of diverse format.

    Parameters
    ----------
    X_csr: CSR Matrix
        Input matrix in CSR format.

    Returns
    -------
    out: iter(Matrices)
        In format['dok', 'lil', 'dia', 'bsr', 'csr', 'csc', 'coo',
        'coo_64', 'csc_64', 'csr_64']
    """
    assert X_csr.format == 'csr'
    yield ('csr', X_csr.copy())
    for sparse_format in ['dok', 'lil', 'dia', 'bsr', 'csc', 'coo']:
        yield (sparse_format, X_csr.asformat(sparse_format))
    X_coo = X_csr.asformat('coo')
    X_coo.row = X_coo.row.astype('int64')
    X_coo.col = X_coo.col.astype('int64')
    yield ('coo_64', X_coo)
    for sparse_format in ['csc', 'csr']:
        X = X_csr.asformat(sparse_format)
        X.indices = X.indices.astype('int64')
        X.indptr = X.indptr.astype('int64')
        yield (sparse_format + '_64', X)