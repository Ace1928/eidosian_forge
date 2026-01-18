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
def test_timedelta_conversion(self):
    f = npdatetime_helpers.get_timedelta_conversion_factor
    for unit in all_units + ('',):
        self.assertEqual(f(unit, unit), 1)
    for unit in all_units:
        self.assertEqual(f('', unit), 1)
    for a, b in itertools.product(time_units, date_units):
        self.assertIs(f(a, b), None)
        self.assertIs(f(b, a), None)

    def check_units_group(group):
        for i, a in enumerate(group):
            for b in group[:i]:
                self.assertGreater(f(b, a), 1, (b, a))
                self.assertIs(f(a, b), None)
    check_units_group(date_units)
    check_units_group(time_units)
    self.assertEqual(f('Y', 'M'), 12)
    self.assertEqual(f('W', 'h'), 24 * 7)
    self.assertEqual(f('W', 'm'), 24 * 7 * 60)
    self.assertEqual(f('W', 'us'), 24 * 7 * 3600 * 1000 * 1000)