from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PlaceSOffsetT(self, x):
    """PlaceSOffsetT prepends a SOffsetT to the Builder, without checking
        for space.
        """
    N.enforce_number(x, N.SOffsetTFlags)
    self.head = self.head - N.SOffsetTFlags.bytewidth
    encode.Write(packer.soffset, self.Bytes, self.Head(), x)