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