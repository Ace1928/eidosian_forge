import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
def _write_fileobject(filename, compress=('zlib', 3)):
    """Return the right compressor file object in write mode."""
    compressmethod = compress[0]
    compresslevel = compress[1]
    if compressmethod in _COMPRESSORS.keys():
        file_instance = _COMPRESSORS[compressmethod].compressor_file(filename, compresslevel=compresslevel)
        return _buffered_write_file(file_instance)
    else:
        file_instance = _COMPRESSORS['zlib'].compressor_file(filename, compresslevel=compresslevel)
        return _buffered_write_file(file_instance)