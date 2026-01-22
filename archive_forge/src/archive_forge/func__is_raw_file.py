import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _is_raw_file(fileobj):
    """Check if fileobj is a raw file object, e.g created with open."""
    fileobj = getattr(fileobj, 'raw', fileobj)
    return isinstance(fileobj, io.FileIO)