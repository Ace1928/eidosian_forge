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
def test_partition_cdtype(self):
    d = np.array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41), ('Lancelot', 1.9, 38)], dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
    tgt = np.sort(d, order=['age', 'height'])
    assert_array_equal(np.partition(d, range(d.size), order=['age', 'height']), tgt)
    assert_array_equal(d[np.argpartition(d, range(d.size), order=['age', 'height'])], tgt)
    for k in range(d.size):
        assert_equal(np.partition(d, k, order=['age', 'height'])[k], tgt[k])
        assert_equal(d[np.argpartition(d, k, order=['age', 'height'])][k], tgt[k])
    d = np.array(['Galahad', 'Arthur', 'zebra', 'Lancelot'])
    tgt = np.sort(d)
    assert_array_equal(np.partition(d, range(d.size)), tgt)
    for k in range(d.size):
        assert_equal(np.partition(d, k)[k], tgt[k])
        assert_equal(d[np.argpartition(d, k)][k], tgt[k])