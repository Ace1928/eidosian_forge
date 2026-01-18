import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_setitem_broadcast(self):
    """
        broadcasted array assignment
        """
    dst = np.arange(5)
    setitem_broadcast_usecase(dst, 42)
    self.assertEqual(dst.tolist(), [42] * 5)
    dst = np.arange(6).reshape(2, 3)
    setitem_broadcast_usecase(dst, np.arange(1, 4))
    self.assertEqual(dst.tolist(), [[1, 2, 3], [1, 2, 3]])
    dst = np.arange(6).reshape(2, 3)
    setitem_broadcast_usecase(dst, np.arange(1, 4).reshape(1, 3))
    self.assertEqual(dst.tolist(), [[1, 2, 3], [1, 2, 3]])
    dst = np.arange(12).reshape(2, 1, 2, 3)
    setitem_broadcast_usecase(dst, np.arange(1, 4).reshape(1, 3))
    inner2 = [[1, 2, 3], [1, 2, 3]]
    self.assertEqual(dst.tolist(), [[inner2]] * 2)
    dst = np.arange(5)
    setitem_broadcast_usecase(dst, np.arange(1, 6).reshape(1, 5))
    self.assertEqual(dst.tolist(), [1, 2, 3, 4, 5])
    dst = np.arange(6).reshape(2, 3)
    setitem_broadcast_usecase(dst, np.arange(1, 1 + dst.size).reshape(1, 1, 2, 3))
    self.assertEqual(dst.tolist(), [[1, 2, 3], [4, 5, 6]])