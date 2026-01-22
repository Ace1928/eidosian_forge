from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PlaceUOffsetT(self, x):
    """PlaceUOffsetT prepends a UOffsetT to the Builder, without checking
        for space.
        """
    N.enforce_number(x, N.UOffsetTFlags)
    self.head = self.head - N.UOffsetTFlags.bytewidth
    encode.Write(packer.uoffset, self.Bytes, self.Head(), x)