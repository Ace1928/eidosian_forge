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
def test_subarray_int_shape(self):
    s = b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n(S'V6'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'|'\np11\nN(S'a'\np12\ng3\ntp13\n(dp14\ng12\n(g7\n(S'V4'\np15\nI0\nI1\ntp16\nRp17\n(I3\nS'|'\np18\n(g7\n(S'i1'\np19\nI0\nI1\ntp20\nRp21\n(I3\nS'|'\np22\nNNNI-1\nI-1\nI0\ntp23\nb(I2\nI2\ntp24\ntp25\nNNI4\nI1\nI0\ntp26\nbI0\ntp27\nsg3\n(g7\n(S'V2'\np28\nI0\nI1\ntp29\nRp30\n(I3\nS'|'\np31\n(g21\nI2\ntp32\nNNI2\nI1\nI0\ntp33\nbI4\ntp34\nsI6\nI1\nI0\ntp35\nbI00\nS'\\x01\\x01\\x01\\x01\\x01\\x02'\np36\ntp37\nb."
    a = np.array([(1, (1, 2))], dtype=[('a', 'i1', (2, 2)), ('b', 'i1', 2)])
    p = self._loads(s)
    assert_equal(a, p)