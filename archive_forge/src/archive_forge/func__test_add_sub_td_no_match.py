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
def _test_add_sub_td_no_match(self, operation):
    """
        Tests that attempting to add/sub a datetime64 and timedelta64
        with types that cannot be cast raises a reasonable exception.
        """

    @njit
    def impl(a, b):
        return operation(a, b)
    fname = operation.__name__
    expected = re.escape(f"ufunc '{fname}' is not supported between datetime64[ns] and timedelta64[M]")
    with self.assertRaisesRegex((TypingError, TypeError), expected):
        impl(np.array([np.datetime64('2011-01-01')], dtype='datetime64[ns]'), np.timedelta64(1000, 'M'))