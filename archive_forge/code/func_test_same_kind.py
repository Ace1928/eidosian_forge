import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def test_same_kind(self):
    f = npdatetime_helpers.same_kind
    for u in all_units:
        self.assertTrue(f(u, u))
    A = ('Y', 'M', 'W', 'D')
    B = ('h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')
    for a, b in itertools.product(A, A):
        self.assertTrue(f(a, b))
    for a, b in itertools.product(B, B):
        self.assertTrue(f(a, b))
    for a, b in itertools.product(A, B):
        self.assertFalse(f(a, b))
        self.assertFalse(f(b, a))