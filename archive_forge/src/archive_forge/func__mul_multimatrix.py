import numpy as np
from ._matrix import spmatrix
from ._base import issparse, _formats, _spbase, sparray
from ._data import _data_matrix
from ._sputils import (
from ._sparsetools import dia_matvec
def _mul_multimatrix(self, other):
    return np.hstack([self._mul_vector(col).reshape(-1, 1) for col in other.T])