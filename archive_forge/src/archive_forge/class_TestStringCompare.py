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
class TestStringCompare:

    def test_string(self):
        g1 = np.array(['This', 'is', 'example'])
        g2 = np.array(['This', 'was', 'example'])
        assert_array_equal(g1 == g2, [g1[i] == g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 != g2, [g1[i] != g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 <= g2, [g1[i] <= g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 >= g2, [g1[i] >= g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 < g2, [g1[i] < g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 > g2, [g1[i] > g2[i] for i in [0, 1, 2]])

    def test_mixed(self):
        g1 = np.array(['spam', 'spa', 'spammer', 'and eggs'])
        g2 = 'spam'
        assert_array_equal(g1 == g2, [x == g2 for x in g1])
        assert_array_equal(g1 != g2, [x != g2 for x in g1])
        assert_array_equal(g1 < g2, [x < g2 for x in g1])
        assert_array_equal(g1 > g2, [x > g2 for x in g1])
        assert_array_equal(g1 <= g2, [x <= g2 for x in g1])
        assert_array_equal(g1 >= g2, [x >= g2 for x in g1])

    def test_unicode(self):
        g1 = np.array(['This', 'is', 'example'])
        g2 = np.array(['This', 'was', 'example'])
        assert_array_equal(g1 == g2, [g1[i] == g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 != g2, [g1[i] != g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 <= g2, [g1[i] <= g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 >= g2, [g1[i] >= g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 < g2, [g1[i] < g2[i] for i in [0, 1, 2]])
        assert_array_equal(g1 > g2, [g1[i] > g2[i] for i in [0, 1, 2]])