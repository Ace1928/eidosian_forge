import numba
from numba.tests.support import TestCase, unittest
from numba.core.registry import cpu_target
from numba.core.compiler import CompilerBase, Flags
from numba.core.compiler_machinery import PassManager
from numba.core import types, ir, bytecode, compiler, ir_utils, registry
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference,
from numba.experimental import jitclass
class Tester(CompilerBase):

    @classmethod
    def mk_pipeline(cls, args, return_type=None, flags=None, locals={}, library=None, typing_context=None, target_context=None):
        if not flags:
            flags = Flags()
        flags.nrt = True
        if typing_context is None:
            typing_context = registry.cpu_target.typing_context
        if target_context is None:
            target_context = registry.cpu_target.target_context
        return cls(typing_context, target_context, library, args, return_type, flags, locals)

    def compile_to_ir(self, func, DCE=False):
        """
                Compile and return IR
                """
        func_id = bytecode.FunctionIdentity.from_function(func)
        self.state.func_id = func_id
        ExtractByteCode().run_pass(self.state)
        state = self.state
        name = 'DCE_testing'
        pm = PassManager(name)
        pm.add_pass(TranslateByteCode, 'analyzing bytecode')
        pm.add_pass(FixupArgs, 'fix up args')
        pm.add_pass(IRProcessing, 'processing IR')
        pm.add_pass(NopythonTypeInference, 'nopython frontend')
        if DCE is True:
            pm.add_pass(DeadCodeElimination, 'DCE after typing')
        pm.finalize()
        pm.run(state)
        return state.func_ir