import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
class TestArrayExpressions(MemoryLeakMixin, TestCase):

    def _compile_function(self, fn, arg_tys):
        """
        Compile the given function both without and with rewrites enabled.
        """
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        cres_0 = control_pipeline.compile_extra(fn)
        control_cfunc = cres_0.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        cres_1 = test_pipeline.compile_extra(fn)
        test_cfunc = cres_1.entry_point
        return (control_pipeline, control_cfunc, test_pipeline, test_cfunc)

    def test_simple_expr(self):
        """
        Using a simple array expression, verify that rewriting is taking
        place, and is fusing loops.
        """
        A = np.linspace(0, 1, 10)
        X = np.linspace(2, 1, 10)
        Y = np.linspace(1, 2, 10)
        arg_tys = [typeof(arg) for arg in (A, X, Y)]
        control_pipeline, nb_axy_0, test_pipeline, nb_axy_1 = self._compile_function(axy, arg_tys)
        control_pipeline2 = RewritesTester.mk_no_rw_pipeline(arg_tys)
        cres_2 = control_pipeline2.compile_extra(ax2)
        nb_ctl = cres_2.entry_point
        expected = nb_axy_0(A, X, Y)
        actual = nb_axy_1(A, X, Y)
        control = nb_ctl(A, X, Y)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(control, actual)
        ir0 = control_pipeline.state.func_ir.blocks
        ir1 = test_pipeline.state.func_ir.blocks
        ir2 = control_pipeline2.state.func_ir.blocks
        self.assertEqual(len(ir0), len(ir1))
        self.assertEqual(len(ir0), len(ir2))
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(ir0[0].body), len(ir2[0].body))

    def _get_array_exprs(self, block):
        for instr in block:
            if isinstance(instr, ir.Assign):
                if isinstance(instr.value, ir.Expr):
                    if instr.value.op == 'arrayexpr':
                        yield instr

    def _array_expr_to_set(self, expr, out=None):
        """
        Convert an array expression tree into a set of operators.
        """
        if out is None:
            out = set()
        if not isinstance(expr, tuple):
            raise ValueError('{0} not a tuple'.format(expr))
        operation, operands = expr
        processed_operands = []
        for operand in operands:
            if isinstance(operand, tuple):
                operand, _ = self._array_expr_to_set(operand, out)
            processed_operands.append(operand)
        processed_expr = (operation, tuple(processed_operands))
        out.add(processed_expr)
        return (processed_expr, out)

    def _test_root_function(self, fn=pos_root):
        A = np.random.random(10)
        B = np.random.random(10) + 1.0
        C = np.random.random(10)
        arg_tys = [typeof(arg) for arg in (A, B, C)]
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        control_cres = control_pipeline.compile_extra(fn)
        nb_fn_0 = control_cres.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        test_cres = test_pipeline.compile_extra(fn)
        nb_fn_1 = test_cres.entry_point
        np_result = fn(A, B, C)
        nb_result_0 = nb_fn_0(A, B, C)
        nb_result_1 = nb_fn_1(A, B, C)
        np.testing.assert_array_almost_equal(np_result, nb_result_0)
        np.testing.assert_array_almost_equal(nb_result_0, nb_result_1)
        return Namespace(locals())

    def _test_cube_function(self, fn=cube):
        A = np.arange(10, dtype=np.float64)
        arg_tys = (typeof(A),)
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys)
        control_cres = control_pipeline.compile_extra(fn)
        nb_fn_0 = control_cres.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys)
        test_cres = test_pipeline.compile_extra(fn)
        nb_fn_1 = test_cres.entry_point
        expected = A ** 3
        self.assertPreciseEqual(expected, nb_fn_0(A))
        self.assertPreciseEqual(expected, nb_fn_1(A))
        return Namespace(locals())

    def _test_explicit_output_function(self, fn):
        """
        Test function having a (a, b, out) signature where *out* is
        an output array the function writes into.
        """
        A = np.arange(10, dtype=np.float64)
        B = A + 1
        arg_tys = (typeof(A),) * 3
        control_pipeline, control_cfunc, test_pipeline, test_cfunc = self._compile_function(fn, arg_tys)

        def run_func(fn):
            out = np.zeros_like(A)
            fn(A, B, out)
            return out
        expected = run_func(fn)
        self.assertPreciseEqual(expected, run_func(control_cfunc))
        self.assertPreciseEqual(expected, run_func(test_cfunc))
        return Namespace(locals())

    def _assert_array_exprs(self, block, expected_count):
        """
        Assert the *block* has the expected number of array expressions
        in it.
        """
        rewrite_count = len(list(self._get_array_exprs(block)))
        self.assertEqual(rewrite_count, expected_count)

    def _assert_total_rewrite(self, control_ir, test_ir, trivial=False):
        """
        Given two dictionaries of Numba IR blocks, check to make sure the
        control IR has no array expressions, while the test IR
        contains one and only one.
        """
        self.assertEqual(len(control_ir), len(test_ir))
        control_block = control_ir[0].body
        test_block = test_ir[0].body
        self._assert_array_exprs(control_block, 0)
        self._assert_array_exprs(test_block, 1)
        if not trivial:
            self.assertGreater(len(control_block), len(test_block))

    def _assert_no_rewrite(self, control_ir, test_ir):
        """
        Given two dictionaries of Numba IR blocks, check to make sure
        the control IR and the test IR both have no array expressions.
        """
        self.assertEqual(len(control_ir), len(test_ir))
        for k, v in control_ir.items():
            control_block = v.body
            test_block = test_ir[k].body
            self.assertEqual(len(control_block), len(test_block))
            self._assert_array_exprs(control_block, 0)
            self._assert_array_exprs(test_block, 0)

    def test_trivial_expr(self):
        """
        Ensure even a non-nested expression is rewritten, as it can enable
        scalar optimizations such as rewriting `x ** 2`.
        """
        ns = self._test_cube_function()
        self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks, trivial=True)

    def test_complicated_expr(self):
        """
        Using the polynomial root function, ensure the full expression is
        being put in the same kernel with no remnants of intermediate
        array expressions.
        """
        ns = self._test_root_function()
        self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)

    def test_common_subexpressions(self, fn=neg_root_common_subexpr):
        """
        Attempt to verify that rewriting will incorporate user common
        subexpressions properly.
        """
        ns = self._test_root_function(fn)
        ir0 = ns.control_pipeline.state.func_ir.blocks
        ir1 = ns.test_pipeline.state.func_ir.blocks
        self.assertEqual(len(ir0), len(ir1))
        self.assertGreater(len(ir0[0].body), len(ir1[0].body))
        self.assertEqual(len(list(self._get_array_exprs(ir0[0].body))), 0)
        array_expr_instrs = list(self._get_array_exprs(ir1[0].body))
        self.assertGreater(len(array_expr_instrs), 1)
        array_sets = list((self._array_expr_to_set(instr.value.expr)[1] for instr in array_expr_instrs))
        for expr_set_0, expr_set_1 in zip(array_sets[:-1], array_sets[1:]):
            intersections = expr_set_0.intersection(expr_set_1)
            if intersections:
                self.fail('Common subexpressions detected in array expressions ({0})'.format(intersections))

    def test_complex_subexpression(self):
        return self.test_common_subexpressions(neg_root_complex_subexpr)

    def test_ufunc_and_dufunc_calls(self):
        """
        Verify that ufunc and DUFunc calls are being properly included in
        array expressions.
        """
        A = np.random.random(10)
        B = np.random.random(10)
        arg_tys = [typeof(arg) for arg in (A, B)]
        vaxy_descr = vaxy._dispatcher.targetdescr
        control_pipeline = RewritesTester.mk_no_rw_pipeline(arg_tys, typing_context=vaxy_descr.typing_context, target_context=vaxy_descr.target_context)
        cres_0 = control_pipeline.compile_extra(call_stuff)
        nb_call_stuff_0 = cres_0.entry_point
        test_pipeline = RewritesTester.mk_pipeline(arg_tys, typing_context=vaxy_descr.typing_context, target_context=vaxy_descr.target_context)
        cres_1 = test_pipeline.compile_extra(call_stuff)
        nb_call_stuff_1 = cres_1.entry_point
        expected = call_stuff(A, B)
        control = nb_call_stuff_0(A, B)
        actual = nb_call_stuff_1(A, B)
        np.testing.assert_array_almost_equal(expected, control)
        np.testing.assert_array_almost_equal(expected, actual)
        self._assert_total_rewrite(control_pipeline.state.func_ir.blocks, test_pipeline.state.func_ir.blocks)

    def test_cmp_op(self):
        """
        Verify that comparison operators are supported by the rewriter.
        """
        ns = self._test_root_function(are_roots_imaginary)
        self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)

    def test_explicit_output(self):
        """
        Check that ufunc calls with explicit outputs are not rewritten.
        """
        ns = self._test_explicit_output_function(explicit_output)
        self._assert_no_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)