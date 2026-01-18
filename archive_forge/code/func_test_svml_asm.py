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
@TestCase.run_test_in_subprocess(envvars=_skylake_axv512_envvars)
def test_svml_asm(self):
    std = '__svml_sin8_ha,'
    fast = '__svml_sin8,'
    self.check(math_sin_loop, 10, what='asm', std_pattern=std, fast_pattern=fast)