from warnings import warn
import operator
import numpy as np
from scipy._lib._util import _prune_array
from ._base import _spbase, issparse, SparseEfficiencyWarning
from ._data import _data_matrix, _minmax_mixin
from . import _sparsetools
from ._sparsetools import (get_csr_submatrix, csr_sample_offsets, csr_todense,
from ._index import IndexMixin
from ._sputils import (upcast, upcast_char, to_native, isdense, isshape,
def _set_self(self, other, copy=False):
    """take the member variables of other and assign them to self"""
    if copy:
        other = other.copy()
    self.data = other.data
    self.indices = other.indices
    self.indptr = other.indptr
    self._shape = check_shape(other.shape)