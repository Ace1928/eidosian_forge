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
def test_ufunc_override_normalize_signature(self):

    class SomeClass:

        def __array_ufunc__(self, ufunc, method, *inputs, **kw):
            return kw
    a = SomeClass()
    kw = np.add(a, [1])
    assert_('sig' not in kw and 'signature' not in kw)
    kw = np.add(a, [1], sig='ii->i')
    assert_('sig' not in kw and 'signature' in kw)
    assert_equal(kw['signature'], 'ii->i')
    kw = np.add(a, [1], signature='ii->i')
    assert_('sig' not in kw and 'signature' in kw)
    assert_equal(kw['signature'], 'ii->i')