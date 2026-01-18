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
def test_redefinition_analysis_different_block_can_exec(self):

    def impl(array, x, a=None):
        b = 0
        if x > 5:
            a = 11
        if x < 4:
            b = 12
        if a is None:
            b += 5
        else:
            b += 7
            if a < 0:
                return 10
        return 30 + b
    self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.float64, types.NoneType('none')), [None, None, None, None], np.zeros((2, 3)), 1.0, None)