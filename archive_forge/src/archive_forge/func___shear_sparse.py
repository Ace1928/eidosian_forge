from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def __shear_sparse(X: scipy.sparse.spmatrix, *, factor: int=+1, axis: int=-1) -> scipy.sparse.spmatrix:
    """Fast shearing for sparse matrices

    Shearing is performed using CSC array indices,
    and the result is converted back to whatever sparse format
    the data was originally provided in.
    """
    fmt = X.format
    if axis == 0:
        X = X.T
    X_shear = X.tocsc(copy=True)
    roll = np.repeat(factor * np.arange(X_shear.shape[1]), np.diff(X_shear.indptr))
    np.mod(X_shear.indices + roll, X_shear.shape[0], out=X_shear.indices)
    if axis == 0:
        X_shear = X_shear.T
    return X_shear.asformat(fmt)