import os
import numpy as np
from .arpack import _arpack  # type: ignore[attr-defined]
from . import eigsh
from scipy._lib._util import check_random_state
from scipy.sparse.linalg._interface import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg  # type: ignore[no-redef]
from scipy.linalg import svd
def _herm(x):
    return x.T.conj()