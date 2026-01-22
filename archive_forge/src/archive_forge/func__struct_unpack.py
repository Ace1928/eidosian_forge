import collections
import gzip
import io
import logging
import struct
import numpy as np
def _struct_unpack(fin, fmt):
    num_bytes = struct.calcsize(fmt)
    return struct.unpack(fmt, fin.read(num_bytes))