from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
class TestPassManagerFunctionality(TestCase):

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

    def test_compiler_error_on_ir_del_from_functionpass(self):
        new_compiler = self._create_pipeline_w_del(FunctionPass, InlineInlinables)

        @njit(pipeline_class=new_compiler)
        def foo(x):
            return x + 1
        with self.assertRaises(errors.CompilerError) as raises:
            foo(10)
        errstr = str(raises.exception)
        self.assertIn('Illegal IR, del found at:', errstr)
        self.assertIn('del x', errstr)

    def test_no_compiler_error_on_ir_del_after_legalization(self):
        new_compiler = self._create_pipeline_w_del(AnalysisPass, IRLegalization)

        @njit(pipeline_class=new_compiler)
        def foo(x):
            return x + 1
        self.assertTrue(foo(10), foo.py_func(10))