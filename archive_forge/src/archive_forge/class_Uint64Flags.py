import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class Uint64Flags(object):
    bytewidth = 8
    min_val = 0
    max_val = 2 ** 64 - 1
    py_type = int
    name = 'uint64'
    packer_type = packer.uint64