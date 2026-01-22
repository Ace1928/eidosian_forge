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
@register_pass(mutates_CFG=False, analysis_only=True)
class SSACheck(AnalysisPass):
    """
            Check SSA on variable `d`
            """
    _name = 'SSA_Check'

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        check(state.func_ir)
        return False