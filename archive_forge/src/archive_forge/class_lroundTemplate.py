import numpy as np
from numba import njit
from numba.core import types, ir
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.tests.support import MemoryLeakMixin, TestCase
@infer_global(issue7507_lround)
class lroundTemplate(AbstractTemplate):
    key = issue7507_lround

    def generic(self, args, kws):
        signature = types.int64(types.float64)

        @lower_builtin(issue7507_lround, types.float64)
        def codegen(context, builder, sig, args):
            return context.cast(builder, args[0], sig.args[0], sig.return_type)
        return signature