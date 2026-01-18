import copy
import os
import random
import re
import io
import sys
import warnings
import gzip
import zlib
import bz2
import pickle
import socket
from contextlib import closing
import mmap
from pathlib import Path
import pytest
from joblib.test.common import np, with_numpy, with_lz4, without_lz4
from joblib.test.common import with_memory_profiler, memory_used
from joblib.testing import parametrize, raises, warns
from joblib import numpy_pickle, register_compressor
from joblib.test import data
from joblib.numpy_pickle_utils import _IO_BUFFER_SIZE
from joblib.numpy_pickle_utils import _detect_compressor
from joblib.numpy_pickle_utils import _is_numpy_array_byte_order_mismatch
from joblib.numpy_pickle_utils import _ensure_native_byte_order
from joblib.compressor import (_COMPRESSORS, _LZ4_PREFIX, CompressorWrapper,
def test_register_compressor_already_registered():
    compressor_name = 'test-name'
    register_compressor(compressor_name, AnotherZlibCompressorWrapper())
    with raises(ValueError) as excinfo:
        register_compressor(compressor_name, StandardLibGzipCompressorWrapper())
    excinfo.match("Compressor '{}' already registered.".format(compressor_name))
    register_compressor(compressor_name, StandardLibGzipCompressorWrapper(), force=True)
    assert compressor_name in _COMPRESSORS
    assert _COMPRESSORS[compressor_name].fileobj_factory == gzip.GzipFile
    _COMPRESSORS.pop(compressor_name)