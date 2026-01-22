from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
class TestFromBuffer:

    @pytest.mark.parametrize('byteorder', ['<', '>'])
    @pytest.mark.parametrize('dtype', [float, int, complex])
    def test_basic(self, byteorder, dtype):
        dt = np.dtype(dtype).newbyteorder(byteorder)
        x = (np.random.random((4, 7)) * 5).astype(dt)
        buf = x.tobytes()
        assert_array_equal(np.frombuffer(buf, dtype=dt), x.flat)

    @pytest.mark.parametrize('obj', [np.arange(10), b'12345678'])
    def test_array_base(self, obj):
        new = np.frombuffer(obj)
        assert new.base is obj

    def test_empty(self):
        assert_array_equal(np.frombuffer(b''), np.array([]))

    @pytest.mark.skipif(IS_PYPY, reason="PyPy's memoryview currently does not track exports. See: https://foss.heptapod.net/pypy/pypy/-/issues/3724")
    def test_mmap_close(self):
        with tempfile.TemporaryFile(mode='wb') as tmp:
            tmp.write(b'asdf')
            tmp.flush()
            mm = mmap.mmap(tmp.fileno(), 0)
            arr = np.frombuffer(mm, dtype=np.uint8)
            with pytest.raises(BufferError):
                mm.close()
            del arr
            mm.close()