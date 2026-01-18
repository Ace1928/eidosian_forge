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
def test_padded_struct_array(self):
    dt1 = np.dtype([('a', 'b'), ('b', 'i'), ('sub', np.dtype('b,i')), ('c', 'i')], align=True)
    x1 = np.arange(dt1.itemsize, dtype=np.int8).view(dt1)
    self._check_roundtrip(x1)
    dt2 = np.dtype([('a', 'b'), ('b', 'i'), ('c', 'b', (3,)), ('d', 'i')], align=True)
    x2 = np.arange(dt2.itemsize, dtype=np.int8).view(dt2)
    self._check_roundtrip(x2)
    dt3 = np.dtype([('a', 'b'), ('b', 'i'), ('c', 'b'), ('d', 'b'), ('e', 'b'), ('sub', np.dtype('b,i', align=True))])
    x3 = np.arange(dt3.itemsize, dtype=np.int8).view(dt3)
    self._check_roundtrip(x3)