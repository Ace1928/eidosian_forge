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
def test_unicode_assignment(self):
    from numpy.core.numeric import set_string_function

    @contextmanager
    def inject_str(s):
        """ replace ndarray.__str__ temporarily """
        set_string_function(lambda x: s, repr=False)
        try:
            yield
        finally:
            set_string_function(None, repr=False)
    a1d = np.array(['test'])
    a0d = np.array('done')
    with inject_str('bad'):
        a1d[0] = a0d
    assert_equal(a1d[0], 'done')
    np.array([np.array('åäö')])