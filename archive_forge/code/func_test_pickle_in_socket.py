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
@with_numpy
def test_pickle_in_socket():
    test_array = np.arange(10)
    _ADDR = ('localhost', 12345)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(_ADDR)
    listener.listen(1)
    with socket.create_connection(_ADDR) as client:
        server, client_addr = listener.accept()
        with server.makefile('wb') as sf:
            numpy_pickle.dump(test_array, sf)
        with client.makefile('rb') as cf:
            array_reloaded = numpy_pickle.load(cf)
        np.testing.assert_array_equal(array_reloaded, test_array)
        bytes_to_send = io.BytesIO()
        numpy_pickle.dump(test_array, bytes_to_send)
        server.send(bytes_to_send.getvalue())
        with client.makefile('rb') as cf:
            array_reloaded = numpy_pickle.load(cf)
        np.testing.assert_array_equal(array_reloaded, test_array)