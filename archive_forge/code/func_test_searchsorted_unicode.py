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
def test_searchsorted_unicode(self):
    a = np.array(['P:\\20x_dapi_cy3\\20x_dapi_cy3_20100185_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100186_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100187_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100189_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100190_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100191_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100192_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100193_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100194_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100195_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100196_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100197_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100198_1', 'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100199_1'], dtype=np.str_)
    ind = np.arange(len(a))
    assert_equal([a.searchsorted(v, 'left') for v in a], ind)
    assert_equal([a.searchsorted(v, 'right') for v in a], ind + 1)
    assert_equal([a.searchsorted(a[i], 'left') for i in ind], ind)
    assert_equal([a.searchsorted(a[i], 'right') for i in ind], ind + 1)