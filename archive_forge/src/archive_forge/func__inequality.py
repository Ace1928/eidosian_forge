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
def _inequality(self, other, op, op_name, bad_scalar_msg):
    if isscalarlike(other):
        if 0 == other and op_name in ('_le_', '_ge_'):
            raise NotImplementedError(" >= and <= don't work with 0.")
        elif op(0, other):
            warn(bad_scalar_msg, SparseEfficiencyWarning, stacklevel=3)
            other_arr = np.empty(self.shape, dtype=np.result_type(other))
            other_arr.fill(other)
            other_arr = self.__class__(other_arr)
            return self._binopt(other_arr, op_name)
        else:
            return self._scalar_binopt(other, op)
    elif isdense(other):
        return op(self.todense(), other)
    elif issparse(other):
        if self.shape != other.shape:
            raise ValueError('inconsistent shapes')
        elif self.format != other.format:
            other = other.asformat(self.format)
        if op_name not in ('_ge_', '_le_'):
            return self._binopt(other, op_name)
        warn('Comparing sparse matrices using >= and <= is inefficient, using <, >, or !=, instead.', SparseEfficiencyWarning, stacklevel=3)
        all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
        res = self._binopt(other, '_gt_' if op_name == '_le_' else '_lt_')
        return all_true - res
    else:
        return NotImplemented