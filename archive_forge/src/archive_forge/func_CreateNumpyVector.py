from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def CreateNumpyVector(self, x):
    """CreateNumpyVector writes a numpy array into the buffer."""
    if np is None:
        raise NumpyRequiredForThisFeature('Numpy was not found.')
    if not isinstance(x, np.ndarray):
        raise TypeError('non-numpy-ndarray passed to CreateNumpyVector')
    if x.dtype.kind not in ['b', 'i', 'u', 'f']:
        raise TypeError('numpy-ndarray holds elements of unsupported datatype')
    if x.ndim > 1:
        raise TypeError('multidimensional-ndarray passed to CreateNumpyVector')
    self.StartVector(x.itemsize, x.size, x.dtype.alignment)
    if x.dtype.str[0] == '<':
        x_lend = x
    else:
        x_lend = x.byteswap(inplace=False)
    l = UOffsetTFlags.py_type(x_lend.itemsize * x_lend.size)
    self.head = UOffsetTFlags.py_type(self.Head() - l)
    self.Bytes[self.Head():self.Head() + l] = x_lend.tobytes(order='C')
    self.vectorNumElems = x.size
    return self.EndVector()