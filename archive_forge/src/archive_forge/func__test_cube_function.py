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