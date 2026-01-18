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
def test_writeable_any_base(self):
    arr = np.arange(10)

    class subclass(np.ndarray):
        pass
    view1 = arr.view(subclass)
    view2 = view1[...]
    arr.flags.writeable = False
    view2.flags.writeable = False
    view2.flags.writeable = True
    arr = np.arange(10)

    class frominterface:

        def __init__(self, arr):
            self.arr = arr
            self.__array_interface__ = arr.__array_interface__
    view1 = np.asarray(frominterface)
    view2 = view1[...]
    view2.flags.writeable = False
    view2.flags.writeable = True
    view1.flags.writeable = False
    view2.flags.writeable = False
    with assert_raises(ValueError):
        view2.flags.writeable = True