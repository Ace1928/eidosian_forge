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
def _scalar_binopt(self, other, op):
    """Scalar version of self._binopt, for cases in which no new nonzeros
        are added. Produces a new sparse array in canonical form.
        """
    self.sum_duplicates()
    res = self._with_data(op(self.data, other), copy=True)
    res.eliminate_zeros()
    return res