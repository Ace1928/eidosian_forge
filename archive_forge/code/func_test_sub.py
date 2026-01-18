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
def test_sub(self):
    f = self.jit(sub_usecase)

    def check(a, b, expected):
        self.assertPreciseEqual(f(a, b), expected)
        self.assertPreciseEqual(f(b, a), -expected)
    check(TD(3), TD(2), TD(1))
    check(TD(3, 's'), TD(2, 's'), TD(1, 's'))
    check(TD(3, 's'), TD(2, 'us'), TD(2999998, 'us'))
    check(TD(1, 'W'), TD(2, 'D'), TD(5, 'D'))
    check(TD('NaT'), TD(1), TD('NaT'))
    check(TD('NaT', 's'), TD(1, 'D'), TD('NaT', 's'))
    check(TD('NaT', 's'), TD(1, 'ms'), TD('NaT', 'ms'))
    with self.assertRaises((TypeError, TypingError)):
        f(TD(1, 'M'), TD(1, 'D'))