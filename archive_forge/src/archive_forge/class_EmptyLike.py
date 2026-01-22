from distutils.version import StrictVersion
import functools
import ast
import numpy as np
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
from .numpy.utils import _STR_2_DTYPE_
from .ndarray.numpy import _internal as _nd_npi
from .symbol.numpy import _internal as _sym_npi
@use_np
class EmptyLike(operator.CustomOp):
    """Fallback to NumPy empty_like operator."""

    def __init__(self, dtype, order, subok, shape):
        super(EmptyLike, self).__init__()
        self._dtype = dtype
        self._order = order
        self._subok = subok
        self._shape = shape

    def forward(self, is_train, req, in_data, out_data, aux):
        np_version = np.version.version
        if StrictVersion(np_version) >= StrictVersion('1.6.0'):
            out = np.empty_like(in_data[0].asnumpy(), dtype=self._dtype, order=self._order, subok=self._subok)
        else:
            out = np.empty_like(in_data[0].asnumpy())
        self.assign(out_data[0], req[0], _mx_np.array(out, ctx=in_data[0].ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError('Operator empty_like does not support gradient computation')