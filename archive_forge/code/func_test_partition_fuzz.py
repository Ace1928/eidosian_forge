import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_partition_fuzz(self):
    pyfunc = partition
    cfunc = jit(nopython=True)(pyfunc)
    for j in range(10, 30):
        for i in range(1, j - 2):
            d = np.arange(j)
            self.rnd.shuffle(d)
            d = d % self.rnd.randint(2, 30)
            idx = self.rnd.randint(d.size)
            kth = [0, idx, i, i + 1, -idx, -i]
            tgt = np.sort(d)[kth]
            self.assertPreciseEqual(cfunc(d, kth)[kth], tgt)
            self.assertPreciseEqual(cfunc(d.tolist(), kth)[kth], tgt)
            self.assertPreciseEqual(cfunc(tuple(d.tolist()), kth)[kth], tgt)
            for k in kth:
                self.partition_sanity_check(pyfunc, cfunc, d, k)