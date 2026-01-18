import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_slicing_1d_broadcast(self):
    dst = np.arange(6).reshape(3, 2)
    src = np.arange(1, 3)
    slicing_1d_usecase_set(dst, src, 0, 2, 1)
    self.assertEqual(dst.tolist(), [[1, 2], [1, 2], [4, 5]])
    dst = np.arange(6).reshape(3, 2)
    src = np.arange(1, 3)
    slicing_1d_usecase_set(dst, src, 0, None, 2)
    self.assertEqual(dst.tolist(), [[1, 2], [2, 3], [1, 2]])
    dst = np.arange(6).reshape(3, 2)
    src = np.arange(1, 5).reshape(2, 2)
    slicing_1d_usecase_set(dst, src, None, 2, 1)
    self.assertEqual(dst.tolist(), [[1, 2], [3, 4], [4, 5]])