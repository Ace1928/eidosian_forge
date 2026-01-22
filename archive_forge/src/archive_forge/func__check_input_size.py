import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .base import (
from .exceptions import DataDimensionalityWarning
from .utils import check_random_state
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.extmath import safe_sparse_dot
from .utils.random import sample_without_replacement
from .utils.validation import check_array, check_is_fitted
def _check_input_size(n_components, n_features):
    """Factorize argument checking for random matrix generation."""
    if n_components <= 0:
        raise ValueError('n_components must be strictly positive, got %d' % n_components)
    if n_features <= 0:
        raise ValueError('n_features must be strictly positive, got %d' % n_features)