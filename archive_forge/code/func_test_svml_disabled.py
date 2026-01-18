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
@TestCase.run_test_in_subprocess(envvars={'NUMBA_DISABLE_INTEL_SVML': '1', **_skylake_axv512_envvars})
def test_svml_disabled(self):

    def math_sin_loop(n):
        ret = np.empty(n, dtype=np.float64)
        for x in range(n):
            ret[x] = math.sin(np.float64(x))
        return ret
    sig = (numba.int32,)
    std = njit(sig)(math_sin_loop)
    fast = njit(sig, fastmath=True)(math_sin_loop)
    fns = (std.overloads[sig], fast.overloads[sig])
    for fn in fns:
        asm = fn.library.get_asm_str()
        self.assertNotIn('__svml_sin', asm)