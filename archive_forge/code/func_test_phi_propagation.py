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
def test_phi_propagation(self):

    @njit
    def foo(actions):
        n = 1
        i = 0
        ct = 0
        while n > 0 and i < len(actions):
            n -= 1
            while actions[i]:
                if actions[i]:
                    if actions[i]:
                        n += 10
                    actions[i] -= 1
                else:
                    if actions[i]:
                        n += 20
                    actions[i] += 1
                ct += n
            ct += n
        return (ct, n)
    self.check_func(foo, np.array([1, 2]))