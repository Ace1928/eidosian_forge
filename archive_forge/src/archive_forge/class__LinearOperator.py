from .base import (
from scipy.sparse.linalg import splu, LinearOperator
from scipy.linalg import eigvals
from scipy.sparse import isspmatrix_csc, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
import numpy as np
from typing import Union, Tuple, Optional, Callable
class _LinearOperator(LinearOperator):

    def __init__(self, matrix: Union[spmatrix, BlockMatrix]):
        self._matrix = matrix
        shape = self._matrix.shape
        dtype = self._matrix.dtype
        super(_LinearOperator, self).__init__(shape=shape, dtype=dtype)

    def _matvec(self, x):
        return self._matrix * x

    def _adjoint(self):
        return _LinearOperator(self._matrix.transpose())