import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
def test_refct_pruning_issue_1511(self):

    @njit
    def f():
        a = np.ones(10, dtype=np.float64)
        b = np.ones(10, dtype=np.float64)
        return (a, b[:])
    a, b = f()
    np.testing.assert_equal(a, b)
    np.testing.assert_equal(a, np.ones(10, dtype=np.float64))