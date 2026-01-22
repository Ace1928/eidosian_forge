import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Int16Flags(object):
    bytewidth = 2
    min_val = -2 ** 15
    max_val = 2 ** 15 - 1
    py_type = int
    name = 'int16'
    packer_type = packer.int16