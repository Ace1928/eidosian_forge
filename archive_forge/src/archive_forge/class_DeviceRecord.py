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
class DeviceRecord(DeviceNDArrayBase):
    """
    An on-GPU record type
    """

    def __init__(self, dtype, stream=0, gpu_data=None):
        shape = ()
        strides = ()
        super(DeviceRecord, self).__init__(shape, strides, dtype, stream, gpu_data)

    @property
    def flags(self):
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
        return dict(self._dummy.flags)

    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
        return numpy_support.from_dtype(self.dtype)

    @devices.require_context
    def __getitem__(self, item):
        return self._do_getitem(item)

    @devices.require_context
    def getitem(self, item, stream=0):
        """Do `__getitem__(item)` with CUDA stream
        """
        return self._do_getitem(item, stream)

    def _do_getitem(self, item, stream=0):
        stream = self._default_stream(stream)
        typ, offset = self.dtype.fields[item]
        newdata = self.gpu_data.view(offset)
        if typ.shape == ():
            if typ.names is not None:
                return DeviceRecord(dtype=typ, stream=stream, gpu_data=newdata)
            else:
                hostary = np.empty(1, dtype=typ)
                _driver.device_to_host(dst=hostary, src=newdata, size=typ.itemsize, stream=stream)
            return hostary[0]
        else:
            shape, strides, dtype = prepare_shape_strides_dtype(typ.shape, None, typ.subdtype[0], 'C')
            return DeviceNDArray(shape=shape, strides=strides, dtype=dtype, gpu_data=newdata, stream=stream)

    @devices.require_context
    def __setitem__(self, key, value):
        return self._do_setitem(key, value)

    @devices.require_context
    def setitem(self, key, value, stream=0):
        """Do `__setitem__(key, value)` with CUDA stream
        """
        return self._do_setitem(key, value, stream=stream)

    def _do_setitem(self, key, value, stream=0):
        stream = self._default_stream(stream)
        synchronous = not stream
        if synchronous:
            ctx = devices.get_context()
            stream = ctx.get_default_stream()
        typ, offset = self.dtype.fields[key]
        newdata = self.gpu_data.view(offset)
        lhs = type(self)(dtype=typ, stream=stream, gpu_data=newdata)
        rhs, _ = auto_device(lhs.dtype.type(value), stream=stream)
        _driver.device_to_device(lhs, rhs, rhs.dtype.itemsize, stream)
        if synchronous:
            stream.synchronize()