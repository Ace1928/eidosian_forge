import string
import warnings
import numpy
import cupy
import cupyx
from cupy import _core
from cupy._core import _scalar
from cupy._creation import basic
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _index
def _arg_minor_reduce(self, ufunc, axis):
    """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.
        Warning: this does not call sum_duplicates()

        Args:
            ufunc (object): Function handle giving the operation to be
                conducted.
            axis (int): Maxtrix over which the reduction should be conducted

        Returns:
            (cupy.ndarray): Reduce result for nonzeros in each
            major_index

        """
    out_shape = self.shape[1 - axis]
    out = cupy.zeros(out_shape, dtype=int)
    ker_name = '_arg_reduction<{}, {}>'.format(_scalar.get_typename(self.data.dtype), _scalar.get_typename(out.dtype))
    if ufunc == cupy.argmax:
        ker = self._max_arg_reduction_mod.get_function('max' + ker_name)
    elif ufunc == cupy.argmin:
        ker = self._min_arg_reduction_mod.get_function('min' + ker_name)
    ker((out_shape,), (1,), (self.data, self.indices, self.indptr[:len(self.indptr) - 1], self.indptr[1:], cupy.int64(self.shape[axis]), out))
    return out