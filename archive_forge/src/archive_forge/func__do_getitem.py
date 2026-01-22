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
def _do_getitem(self, item, stream=0):
    stream = self._default_stream(stream)
    arr = self._dummy.__getitem__(item)
    extents = list(arr.iter_contiguous_extent())
    cls = type(self)
    if len(extents) == 1:
        newdata = self.gpu_data.view(*extents[0])
        if not arr.is_array:
            if self.dtype.names is not None:
                return DeviceRecord(dtype=self.dtype, stream=stream, gpu_data=newdata)
            else:
                hostary = np.empty(1, dtype=self.dtype)
                _driver.device_to_host(dst=hostary, src=newdata, size=self._dummy.itemsize, stream=stream)
            return hostary[0]
        else:
            return cls(shape=arr.shape, strides=arr.strides, dtype=self.dtype, gpu_data=newdata, stream=stream)
    else:
        newdata = self.gpu_data.view(*arr.extent)
        return cls(shape=arr.shape, strides=arr.strides, dtype=self.dtype, gpu_data=newdata, stream=stream)