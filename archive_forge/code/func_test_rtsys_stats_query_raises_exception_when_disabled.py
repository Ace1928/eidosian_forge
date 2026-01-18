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
def test_rtsys_stats_query_raises_exception_when_disabled(self):
    _nrt_python.memsys_disable_stats()
    self.assertFalse(_nrt_python.memsys_stats_enabled())
    with self.assertRaises(RuntimeError) as raises:
        rtsys.get_allocation_stats()
    self.assertIn('NRT stats are disabled.', str(raises.exception))