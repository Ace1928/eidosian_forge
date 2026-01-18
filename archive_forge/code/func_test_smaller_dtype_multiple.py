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
def test_smaller_dtype_multiple(self):
    x = np.arange(10, dtype='<i4')[::2]
    with pytest.raises(ValueError, match='the last axis must be contiguous'):
        x.view('<i2')
    expected = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]]
    assert_array_equal(x[:, np.newaxis].view('<i2'), expected)