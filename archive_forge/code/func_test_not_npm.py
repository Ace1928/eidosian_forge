import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def test_not_npm(self):
    pyfunc = self.op.not_usecase
    argtys = [types.int8, types.int32, types.int64, types.float32, types.complex128]
    values = [1, 2, 3, 1.2, 3.4j]
    for ty, val in zip(argtys, values):
        cfunc = njit((ty,))(pyfunc)
        self.assertEqual(cfunc.nopython_signatures[0].return_type, types.boolean)
        self.assertEqual(pyfunc(val), cfunc(val))