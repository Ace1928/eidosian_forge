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
@njit(pipeline_class=CustomPipeline)
def while_for(n, max_iter=1):
    a = np.empty((n, n))
    i = 0
    while i <= max_iter:
        for j in range(len(a)):
            for k in range(len(a)):
                a[j, k] = j + k
        i += 1
    return a