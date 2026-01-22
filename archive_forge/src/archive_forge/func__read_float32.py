import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_float32(f):
    """Read a 32-bit float"""
    return np.float32(struct.unpack('>f', f.read(4))[0])