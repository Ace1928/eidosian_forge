import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
def valid_number(n, flags):
    if flags.min_val is None and flags.max_val is None:
        return True
    return flags.min_val <= n <= flags.max_val