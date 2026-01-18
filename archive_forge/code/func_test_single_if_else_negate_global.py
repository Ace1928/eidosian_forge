import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
def test_single_if_else_negate_global(self):
    global c_test_single_if_else_negate_global
    for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
        for c in c_inp:
            c_test_single_if_else_negate_global = c

            def func(x):
                if not c_test_single_if_else_negate_global:
                    return (3.14159, c_test_single_if_else_negate_global)
                else:
                    return (1.61803, c_test_single_if_else_negate_global)
            self.assert_prune(func, (types.NoneType('none'),), [prune], None)