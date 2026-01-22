from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from ..base import (
from ..utils import check_array, check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import check_is_fitted
Number of transformed output features.