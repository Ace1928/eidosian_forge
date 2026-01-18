import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
def test_omitted_args(self):
    distinct = DistinctChecker()
    v0 = OmittedArg(0.0)
    v1 = OmittedArg(1.0)
    v2 = OmittedArg(1)
    s = compute_fingerprint(v0)
    self.assertEqual(compute_fingerprint(v1), s)
    distinct.add(s)
    distinct.add(compute_fingerprint(v2))
    distinct.add(compute_fingerprint(0.0))
    distinct.add(compute_fingerprint(1))