import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
def _check_dump_func_opt_llvm(self, out):
    self.assertIn('--FUNCTION OPTIMIZED DUMP %s' % self.func_name, out)
    self.assertIn('add nsw i64 %arg.somearg, 1', out)