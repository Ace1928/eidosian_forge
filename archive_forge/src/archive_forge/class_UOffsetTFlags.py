import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class UOffsetTFlags(Uint32Flags):
    pass