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
def test_undefined_var(self):
    with override_config('ALWAYS_WARN_UNINIT_VAR', 0):
        self.check_undefined_var(should_warn=False)
    with override_config('ALWAYS_WARN_UNINIT_VAR', 1):
        self.check_undefined_var(should_warn=True)