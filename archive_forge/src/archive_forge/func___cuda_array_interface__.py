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
@property
def __cuda_array_interface__(self):
    if _driver.USE_NV_BINDING:
        if self.device_ctypes_pointer is not None:
            ptr = int(self.device_ctypes_pointer)
        else:
            ptr = 0
    elif self.device_ctypes_pointer.value is not None:
        ptr = self.device_ctypes_pointer.value
    else:
        ptr = 0
    return {'shape': tuple(self.shape), 'strides': None if is_contiguous(self) else tuple(self.strides), 'data': (ptr, False), 'typestr': self.dtype.str, 'stream': int(self.stream) if self.stream != 0 else None, 'version': 3}