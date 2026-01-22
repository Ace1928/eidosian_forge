import sys
import gc
import gzip
import os
import threading
import time
import warnings
import io
import re
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
from datetime import datetime
import locale
from multiprocessing import Value, get_context
from ctypes import c_bool
import numpy as np
import numpy.ma as ma
from numpy.lib._iotools import ConverterError, ConversionWarning
from numpy.compat import asbytes
from numpy.ma.testutils import assert_equal
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
class TestPathUsage:

    def test_loadtxt(self):
        with temppath(suffix='.txt') as path:
            path = Path(path)
            a = np.array([[1.1, 2], [3, 4]])
            np.savetxt(path, a)
            x = np.loadtxt(path)
            assert_array_equal(x, a)

    def test_save_load(self):
        with temppath(suffix='.npy') as path:
            path = Path(path)
            a = np.array([[1, 2], [3, 4]], int)
            np.save(path, a)
            data = np.load(path)
            assert_array_equal(data, a)

    def test_save_load_memmap(self):
        with temppath(suffix='.npy') as path:
            path = Path(path)
            a = np.array([[1, 2], [3, 4]], int)
            np.save(path, a)
            data = np.load(path, mmap_mode='r')
            assert_array_equal(data, a)
            del data
            if IS_PYPY:
                break_cycles()
                break_cycles()

    @pytest.mark.xfail(IS_WASM, reason="memmap doesn't work correctly")
    def test_save_load_memmap_readwrite(self):
        with temppath(suffix='.npy') as path:
            path = Path(path)
            a = np.array([[1, 2], [3, 4]], int)
            np.save(path, a)
            b = np.load(path, mmap_mode='r+')
            a[0][0] = 5
            b[0][0] = 5
            del b
            if IS_PYPY:
                break_cycles()
                break_cycles()
            data = np.load(path)
            assert_array_equal(data, a)

    def test_savez_load(self):
        with temppath(suffix='.npz') as path:
            path = Path(path)
            np.savez(path, lab='place holder')
            with np.load(path) as data:
                assert_array_equal(data['lab'], 'place holder')

    def test_savez_compressed_load(self):
        with temppath(suffix='.npz') as path:
            path = Path(path)
            np.savez_compressed(path, lab='place holder')
            data = np.load(path)
            assert_array_equal(data['lab'], 'place holder')
            data.close()

    def test_genfromtxt(self):
        with temppath(suffix='.txt') as path:
            path = Path(path)
            a = np.array([(1, 2), (3, 4)])
            np.savetxt(path, a)
            data = np.genfromtxt(path)
            assert_array_equal(a, data)

    def test_recfromtxt(self):
        with temppath(suffix='.txt') as path:
            path = Path(path)
            with path.open('w') as f:
                f.write('A,B\n0,1\n2,3')
            kwargs = dict(delimiter=',', missing_values='N/A', names=True)
            test = np.recfromtxt(path, **kwargs)
            control = np.array([(0, 1), (2, 3)], dtype=[('A', int), ('B', int)])
            assert_(isinstance(test, np.recarray))
            assert_equal(test, control)

    def test_recfromcsv(self):
        with temppath(suffix='.txt') as path:
            path = Path(path)
            with path.open('w') as f:
                f.write('A,B\n0,1\n2,3')
            kwargs = dict(missing_values='N/A', names=True, case_sensitive=True)
            test = np.recfromcsv(path, dtype=None, **kwargs)
            control = np.array([(0, 1), (2, 3)], dtype=[('A', int), ('B', int)])
            assert_(isinstance(test, np.recarray))
            assert_equal(test, control)