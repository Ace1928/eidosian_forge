import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_setitem_readonly(self):
    arr = np.arange(5)
    arr.flags.writeable = False
    with self.assertRaises((TypeError, errors.TypingError)) as raises:
        setitem_usecase(arr, 1, 42)
    self.assertIn('Cannot modify readonly array of type:', str(raises.exception))