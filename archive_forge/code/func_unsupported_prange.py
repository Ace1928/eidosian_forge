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
def unsupported_prange(n):
    a = np.ones(n)
    for i in prange(n):
        a[i] = a[i] + np.sin(i)
        assert i + 13 < 100000
    return a