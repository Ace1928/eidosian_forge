import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
class SSABaseTest(TestCase):

    def check_func(self, func, *args):
        got = func(*copy.deepcopy(args))
        exp = func.py_func(*copy.deepcopy(args))
        self.assertEqual(got, exp)