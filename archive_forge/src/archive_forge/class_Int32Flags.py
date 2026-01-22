import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Int32Flags(object):
    bytewidth = 4
    min_val = -2 ** 31
    max_val = 2 ** 31 - 1
    py_type = int
    name = 'int32'
    packer_type = packer.int32