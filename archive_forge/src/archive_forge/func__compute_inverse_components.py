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
def _compute_inverse_components(self):
    """Compute the pseudo-inverse of the (densified) components."""
    components = self.components_
    if sp.issparse(components):
        components = components.toarray()
    return linalg.pinv(components, check_finite=False)