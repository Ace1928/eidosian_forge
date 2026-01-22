import math
import functools
import operator
import copy
from ctypes import c_void_p
import numpy as np
import numba
from numba import _devicearray
from numba.cuda.cudadrv import devices
from numba.cuda.cudadrv import driver as _driver
from numba.core import types, config
from numba.np.unsafe.ndarray import to_fixed_tuple
from numba.misc import dummyarray
from numba.np import numpy_support
from numba.cuda.api_util import prepare_shape_strides_dtype
from numba.core.errors import NumbaPerformanceWarning
from warnings import warn
def _do_setitem(self, key, value, stream=0):
    stream = self._default_stream(stream)
    synchronous = not stream
    if synchronous:
        ctx = devices.get_context()
        stream = ctx.get_default_stream()
    arr = self._dummy.__getitem__(key)
    newdata = self.gpu_data.view(*arr.extent)
    if isinstance(arr, dummyarray.Element):
        shape = ()
        strides = ()
    else:
        shape = arr.shape
        strides = arr.strides
    lhs = type(self)(shape=shape, strides=strides, dtype=self.dtype, gpu_data=newdata, stream=stream)
    rhs, _ = auto_device(value, stream=stream, user_explicit=True)
    if rhs.ndim > lhs.ndim:
        raise ValueError("Can't assign %s-D array to %s-D self" % (rhs.ndim, lhs.ndim))
    rhs_shape = np.ones(lhs.ndim, dtype=np.int64)
    rhs_shape[lhs.ndim - rhs.ndim:] = rhs.shape
    rhs = rhs.reshape(*rhs_shape)
    for i, (l, r) in enumerate(zip(lhs.shape, rhs.shape)):
        if r != 1 and l != r:
            raise ValueError("Can't copy sequence with size %d to array axis %d with dimension %d" % (r, i, l))
    n_elements = functools.reduce(operator.mul, lhs.shape, 1)
    _assign_kernel(lhs.ndim).forall(n_elements, stream=stream)(lhs, rhs)
    if synchronous:
        stream.synchronize()