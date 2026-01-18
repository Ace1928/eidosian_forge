import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
Read from file-like object until size bytes are read.

    TODO python2_drop: is it still needed? The docstring mentions python 2.6
    and it looks like this can be at least simplified ...

    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.

    This function was taken from numpy/lib/format.py in version 1.10.2.

    Parameters
    ----------
    fp: file-like object
    size: int
    error_template: str

    Returns
    -------
    a bytes object
        The data read in bytes.

    