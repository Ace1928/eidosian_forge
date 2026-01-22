from warnings import warn
import numpy as np
from numpy import asarray
from scipy.sparse import (issparse,
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.linalg import LinAlgError
import copy
from . import _superlu
def _safe_downcast_indices(A):
    max_value = np.iinfo(np.intc).max
    if A.indptr[-1] > max_value:
        raise ValueError('indptr values too large for SuperLU')
    if max(*A.shape) > max_value:
        if np.any(A.indices > max_value):
            raise ValueError('indices values too large for SuperLU')
    indices = A.indices.astype(np.intc, copy=False)
    indptr = A.indptr.astype(np.intc, copy=False)
    return (indices, indptr)