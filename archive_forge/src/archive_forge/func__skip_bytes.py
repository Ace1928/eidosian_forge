import struct
import numpy as np
import tempfile
import zlib
import warnings
def _skip_bytes(f, n):
    """Skip `n` bytes"""
    f.read(n)
    return