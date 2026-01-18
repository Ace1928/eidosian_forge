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
@pytest.mark.parametrize(['flag', 'flag_value', 'writeable'], [('writeable', True, True), ('_warn_on_write', True, False), ('writeable', False, False)])
def test_readonly_flag_protocols(self, flag, flag_value, writeable):
    a = np.arange(10)
    setattr(a.flags, flag, flag_value)

    class MyArr:
        __array_struct__ = a.__array_struct__
    assert memoryview(a).readonly is not writeable
    assert a.__array_interface__['data'][1] is not writeable
    assert np.asarray(MyArr()).flags.writeable is writeable