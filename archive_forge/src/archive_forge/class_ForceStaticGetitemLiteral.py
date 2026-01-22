import numpy as np
from numba import njit
from numba.core import types, ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.tests.support import MemoryLeakMixin, TestCase
@register_pass(mutates_CFG=False, analysis_only=False)
class ForceStaticGetitemLiteral(FunctionPass):
    _name = 'force_static_getitem_literal'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        repl = {}
        for inst, sig in state.calltypes.items():
            if isinstance(inst, ir.Expr) and inst.op == 'static_getitem':
                [obj, idx] = sig.args
                new_sig = sig.replace(args=(obj, types.literal(inst.index)))
                repl[inst] = new_sig
        state.calltypes.update(repl)
        return True