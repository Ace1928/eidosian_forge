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
class TestFlat:

    def setup_method(self):
        a0 = np.arange(20.0)
        a = a0.reshape(4, 5)
        a0.shape = (4, 5)
        a.flags.writeable = False
        self.a = a
        self.b = a[::2, ::2]
        self.a0 = a0
        self.b0 = a0[::2, ::2]

    def test_contiguous(self):
        testpassed = False
        try:
            self.a.flat[12] = 100.0
        except ValueError:
            testpassed = True
        assert_(testpassed)
        assert_(self.a.flat[12] == 12.0)

    def test_discontiguous(self):
        testpassed = False
        try:
            self.b.flat[4] = 100.0
        except ValueError:
            testpassed = True
        assert_(testpassed)
        assert_(self.b.flat[4] == 12.0)

    def test___array__(self):
        c = self.a.flat.__array__()
        d = self.b.flat.__array__()
        e = self.a0.flat.__array__()
        f = self.b0.flat.__array__()
        assert_(c.flags.writeable is False)
        assert_(d.flags.writeable is False)
        assert_(e.flags.writeable is True)
        assert_(f.flags.writeable is False)
        assert_(c.flags.writebackifcopy is False)
        assert_(d.flags.writebackifcopy is False)
        assert_(e.flags.writebackifcopy is False)
        assert_(f.flags.writebackifcopy is False)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
    def test_refcount(self):
        inds = [np.intp(0), np.array([True] * self.a.size), np.array([0]), None]
        indtype = np.dtype(np.intp)
        rc_indtype = sys.getrefcount(indtype)
        for ind in inds:
            rc_ind = sys.getrefcount(ind)
            for _ in range(100):
                try:
                    self.a.flat[ind]
                except IndexError:
                    pass
            assert_(abs(sys.getrefcount(ind) - rc_ind) < 50)
            assert_(abs(sys.getrefcount(indtype) - rc_indtype) < 50)

    def test_index_getset(self):
        it = np.arange(10).reshape(2, 1, 5).flat
        with pytest.raises(AttributeError):
            it.index = 10
        for _ in it:
            pass
        assert it.index == it.base.size