from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def Prep(self, size, additionalBytes):
    """
        Prep prepares to write an element of `size` after `additional_bytes`
        have been written, e.g. if you write a string, you need to align
        such the int length field is aligned to SizeInt32, and the string
        data follows it directly.
        If all you need to do is align, `additionalBytes` will be 0.
        """
    if size > self.minalign:
        self.minalign = size
    alignSize = ~(len(self.Bytes) - self.Head() + additionalBytes) + 1
    alignSize &= size - 1
    while self.Head() < alignSize + size + additionalBytes:
        oldBufSize = len(self.Bytes)
        self.growByteBuffer()
        updated_head = self.head + len(self.Bytes) - oldBufSize
        self.head = UOffsetTFlags.py_type(updated_head)
    self.Pad(alignSize)