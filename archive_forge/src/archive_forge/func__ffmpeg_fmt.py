from __future__ import absolute_import, division, print_function
import errno
import os
import subprocess
import sys
import tempfile
import numpy as np
from ..utils import string_types
from ..audio.signal import Signal
def _ffmpeg_fmt(dtype):
    """
    Convert numpy dtypes to format strings understood by ffmpeg.

    Parameters
    ----------
    dtype : numpy dtype
        Data type to be converted.

    Returns
    -------
    str
        ffmpeg format string.

    """
    dtype = np.dtype(dtype)
    fmt = {'u': 'u', 'i': 's', 'f': 'f'}.get(dtype.kind)
    fmt += str(8 * dtype.itemsize)
    if dtype.byteorder == '=':
        fmt += sys.byteorder[0] + 'e'
    else:
        fmt += {'|': '', '<': 'le', '>': 'be'}.get(dtype.byteorder)
    return str(fmt)