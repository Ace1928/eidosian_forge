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
def test_single_if_negate_freevar(self):
    for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
        for const in c_inp:

            def func(x):
                if not const:
                    return (3.14159, const)
            self.assert_prune(func, (types.NoneType('none'),), [prune], None)