import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Float32Flags(object):
    bytewidth = 4
    min_val = None
    max_val = None
    py_type = float
    name = 'float32'
    packer_type = packer.float32