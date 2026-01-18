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
def test_scalar_context_asm(self):
    pat = '$_sin' if config.IS_OSX else '$sin'
    self.check(math_sin_scalar, 7.0, what='asm', std_pattern=pat)
    self.check(math_sin_scalar, 7.0, what='asm', fast_pattern=pat)