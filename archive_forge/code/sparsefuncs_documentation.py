import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
Create an implicitly offset linear operator.

    This is used by PCA on sparse data to avoid densifying the whole data
    matrix.

    Params
    ------
        X : sparse matrix of shape (n_samples, n_features)
        offset : ndarray of shape (n_features,)

    Returns
    -------
    centered : LinearOperator
    