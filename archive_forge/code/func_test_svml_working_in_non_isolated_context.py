import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
def test_svml_working_in_non_isolated_context(self):

    @njit(fastmath={'fast'}, error_model='numpy')
    def impl(n):
        x = np.empty(n * 8, dtype=np.float64)
        ret = np.empty_like(x)
        for i in range(ret.size):
            ret[i] += math.cosh(x[i])
        return ret
    impl(1)
    self.assertTrue('intel_svmlcc' in impl.inspect_llvm(impl.signatures[0]))