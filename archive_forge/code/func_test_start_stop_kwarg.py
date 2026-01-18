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
def test_start_stop_kwarg(self):
    keyword_stop = np.arange(stop=3)
    keyword_zerotostop = np.arange(start=0, stop=3)
    keyword_start_stop = np.arange(start=3, stop=9)
    assert len(keyword_stop) == 3
    assert len(keyword_zerotostop) == 3
    assert len(keyword_start_stop) == 6
    assert_array_equal(keyword_stop, keyword_zerotostop)