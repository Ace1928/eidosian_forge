import re
import numpy as np
from numba.tests.support import (TestCase, override_config, captured_stdout,
from numba import jit, njit
from numba.core import types, ir, postproc, compiler
from numba.core.ir_utils import (guard, find_callname, find_const,
from numba.core.registry import CPUDispatcher
from numba.core.inline_closurecall import inline_closure_call
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
import unittest
class TestInlining(TestCase):
    """
    Check that jitted inner functions are inlined into outer functions,
    in nopython mode.
    Note that not all inner functions are guaranteed to be inlined.
    We just trust LLVM's inlining heuristics.
    """

    def make_pattern(self, fullname):
        """
        Make regexpr to match mangled name
        """
        parts = fullname.split('.')
        return '_ZN?' + ''.join(['\\d+{}'.format(p) for p in parts])

    def assert_has_pattern(self, fullname, text):
        pat = self.make_pattern(fullname)
        self.assertIsNotNone(re.search(pat, text), msg='expected {}'.format(pat))

    def assert_not_has_pattern(self, fullname, text):
        pat = self.make_pattern(fullname)
        self.assertIsNone(re.search(pat, text), msg='unexpected {}'.format(pat))

    def test_inner_function(self):
        from numba.tests.inlining_usecases import outer_simple, __name__ as prefix
        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_simple)
        self.assertPreciseEqual(cfunc(1), 4)
        asm = out.getvalue()
        self.assert_has_pattern('%s.outer_simple' % prefix, asm)
        self.assert_not_has_pattern('%s.inner' % prefix, asm)

    def test_multiple_inner_functions(self):
        from numba.tests.inlining_usecases import outer_multiple, __name__ as prefix
        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_multiple)
        self.assertPreciseEqual(cfunc(1), 6)
        asm = out.getvalue()
        self.assert_has_pattern('%s.outer_multiple' % prefix, asm)
        self.assert_not_has_pattern('%s.more' % prefix, asm)
        self.assert_not_has_pattern('%s.inner' % prefix, asm)

    @skip_parfors_unsupported
    def test_inline_call_after_parfor(self):
        from numba.tests.inlining_usecases import __dummy__

        def test_impl(A):
            __dummy__()
            return A.sum()
        j_func = njit(parallel=True, pipeline_class=InlineTestPipeline)(test_impl)
        A = np.arange(10)
        self.assertEqual(test_impl(A), j_func(A))

    @skip_parfors_unsupported
    def test_inline_update_target_def(self):

        def test_impl(a):
            if a == 1:
                b = 2
            else:
                b = 3
            return b
        func_ir = compiler.run_frontend(test_impl)
        blocks = list(func_ir.blocks.values())
        for block in blocks:
            for i, stmt in enumerate(block.body):
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var) and (guard(find_const, func_ir, stmt.value) == 2):
                    func_ir._definitions[stmt.target.name].remove(stmt.value)
                    stmt.value = ir.Expr.call(ir.Var(block.scope, 'myvar', loc=stmt.loc), (), (), stmt.loc)
                    func_ir._definitions[stmt.target.name].append(stmt.value)
                    inline_closure_call(func_ir, {}, block, i, lambda: 2)
                    break
        self.assertEqual(len(func_ir._definitions['b']), 2)

    @skip_parfors_unsupported
    def test_inline_var_dict_ret(self):

        @njit(locals={'b': types.float64})
        def g(a):
            b = a + 1
            return b

        def test_impl():
            return g(1)
        func_ir = compiler.run_frontend(test_impl)
        blocks = list(func_ir.blocks.values())
        for block in blocks:
            for i, stmt in enumerate(block.body):
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'call'):
                    func_def = guard(get_definition, func_ir, stmt.value.func)
                    if isinstance(func_def, (ir.Global, ir.FreeVar)) and isinstance(func_def.value, CPUDispatcher):
                        py_func = func_def.value.py_func
                        _, var_map = inline_closure_call(func_ir, py_func.__globals__, block, i, py_func)
                        break
        self.assertTrue('b' in var_map)

    @skip_parfors_unsupported
    def test_inline_call_branch_pruning(self):

        @njit
        def foo(A=None):
            if A is None:
                return 2
            else:
                return A

        def test_impl(A=None):
            return foo(A)

        @register_pass(analysis_only=False, mutates_CFG=True)
        class PruningInlineTestPass(FunctionPass):
            _name = 'pruning_inline_test_pass'

            def __init__(self):
                FunctionPass.__init__(self)

            def run_pass(self, state):
                assert len(state.func_ir.blocks) == 1
                block = list(state.func_ir.blocks.values())[0]
                for i, stmt in enumerate(block.body):
                    if guard(find_callname, state.func_ir, stmt.value) is not None:
                        inline_closure_call(state.func_ir, {}, block, i, foo.py_func, state.typingctx, state.targetctx, (state.typemap[stmt.value.args[0].name],), state.typemap, state.calltypes)
                        break
                return True

        class InlineTestPipelinePrune(compiler.CompilerBase):

            def define_pipelines(self):
                pm = gen_pipeline(self.state, PruningInlineTestPass)
                pm.finalize()
                return [pm]
        j_func = njit(pipeline_class=InlineTestPipelinePrune)(test_impl)
        A = 3
        self.assertEqual(test_impl(A), j_func(A))
        self.assertEqual(test_impl(), j_func())
        fir = j_func.overloads[types.Omitted(None),].metadata['preserved_ir']
        fir.blocks = simplify_CFG(fir.blocks)
        self.assertEqual(len(fir.blocks), 1)