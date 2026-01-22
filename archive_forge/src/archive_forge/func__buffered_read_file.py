import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _buffered_read_file(fobj):
    """Return a buffered version of a read file object."""
    return io.BufferedReader(fobj, buffer_size=_IO_BUFFER_SIZE)