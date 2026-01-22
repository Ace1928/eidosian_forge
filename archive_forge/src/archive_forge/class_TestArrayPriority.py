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
class TestArrayPriority:
    op = operator
    binary_ops = [op.pow, op.add, op.sub, op.mul, op.floordiv, op.truediv, op.mod, op.and_, op.or_, op.xor, op.lshift, op.rshift, op.mod, op.gt, op.ge, op.lt, op.le, op.ne, op.eq]

    class Foo(np.ndarray):
        __array_priority__ = 100.0

        def __new__(cls, *args, **kwargs):
            return np.array(*args, **kwargs).view(cls)

    class Bar(np.ndarray):
        __array_priority__ = 101.0

        def __new__(cls, *args, **kwargs):
            return np.array(*args, **kwargs).view(cls)

    class Other:
        __array_priority__ = 1000.0

        def _all(self, other):
            return self.__class__()
        __add__ = __radd__ = _all
        __sub__ = __rsub__ = _all
        __mul__ = __rmul__ = _all
        __pow__ = __rpow__ = _all
        __div__ = __rdiv__ = _all
        __mod__ = __rmod__ = _all
        __truediv__ = __rtruediv__ = _all
        __floordiv__ = __rfloordiv__ = _all
        __and__ = __rand__ = _all
        __xor__ = __rxor__ = _all
        __or__ = __ror__ = _all
        __lshift__ = __rlshift__ = _all
        __rshift__ = __rrshift__ = _all
        __eq__ = _all
        __ne__ = _all
        __gt__ = _all
        __ge__ = _all
        __lt__ = _all
        __le__ = _all

    def test_ndarray_subclass(self):
        a = np.array([1, 2])
        b = self.Bar([1, 2])
        for f in self.binary_ops:
            msg = repr(f)
            assert_(isinstance(f(a, b), self.Bar), msg)
            assert_(isinstance(f(b, a), self.Bar), msg)

    def test_ndarray_other(self):
        a = np.array([1, 2])
        b = self.Other()
        for f in self.binary_ops:
            msg = repr(f)
            assert_(isinstance(f(a, b), self.Other), msg)
            assert_(isinstance(f(b, a), self.Other), msg)

    def test_subclass_subclass(self):
        a = self.Foo([1, 2])
        b = self.Bar([1, 2])
        for f in self.binary_ops:
            msg = repr(f)
            assert_(isinstance(f(a, b), self.Bar), msg)
            assert_(isinstance(f(b, a), self.Bar), msg)

    def test_subclass_other(self):
        a = self.Foo([1, 2])
        b = self.Other()
        for f in self.binary_ops:
            msg = repr(f)
            assert_(isinstance(f(a, b), self.Other), msg)
            assert_(isinstance(f(b, a), self.Other), msg)