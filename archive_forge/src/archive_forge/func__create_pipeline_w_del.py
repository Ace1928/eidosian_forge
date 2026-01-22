from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
def _create_pipeline_w_del(self, base=None, inject_after=None):
    """
        Creates a new compiler pipeline with the _InjectDelsPass injected after
        the pass supplied in kwarg 'inject_after'.
        """
    self.assertTrue(inject_after is not None)
    self.assertTrue(base is not None)

    @register_pass(mutates_CFG=False, analysis_only=False)
    class _InjectDelsPass(base):
        """
            This pass injects ir.Del nodes into the IR
            """
        _name = 'inject_dels_%s' % str(base)

        def __init__(self):
            base.__init__(self)

        def run_pass(self, state):
            pp = postproc.PostProcessor(state.func_ir)
            pp.run(emit_dels=True)
            return True

    class TestCompiler(Compiler):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(_InjectDelsPass, inject_after)
            pm.finalize()
            return [pm]
    return TestCompiler