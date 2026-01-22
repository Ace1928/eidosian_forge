import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_uint32(f):
    """Read an unsigned 32-bit integer"""
    return np.uint32(struct.unpack('>I', f.read(4))[0])