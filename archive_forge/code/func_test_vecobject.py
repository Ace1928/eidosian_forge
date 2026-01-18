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
def test_vecobject(self):

    class Vec:

        def __init__(self, sequence=None):
            if sequence is None:
                sequence = []
            self.array = np.array(sequence)

        def __add__(self, other):
            out = Vec()
            out.array = self.array + other.array
            return out

        def __sub__(self, other):
            out = Vec()
            out.array = self.array - other.array
            return out

        def __mul__(self, other):
            out = Vec(self.array.copy())
            out.array *= other
            return out

        def __rmul__(self, other):
            return self * other
    U_non_cont = np.transpose([[1.0, 1.0], [1.0, 2.0]])
    U_cont = np.ascontiguousarray(U_non_cont)
    x = np.array([Vec([1.0, 0.0]), Vec([0.0, 1.0])])
    zeros = np.array([Vec([0.0, 0.0]), Vec([0.0, 0.0])])
    zeros_test = np.dot(U_cont, x) - np.dot(U_non_cont, x)
    assert_equal(zeros[0].array, zeros_test[0].array)
    assert_equal(zeros[1].array, zeros_test[1].array)