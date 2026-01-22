import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
class BoolFlags(object):
    bytewidth = 1
    min_val = False
    max_val = True
    py_type = bool
    name = 'bool'
    packer_type = packer.boolean