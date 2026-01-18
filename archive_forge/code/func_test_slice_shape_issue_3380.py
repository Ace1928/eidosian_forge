import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
@skip_unsupported
def test_slice_shape_issue_3380(self):

    def test_impl1():
        a = slice(None, None)
        return True
    self.assertEqual(njit(test_impl1, parallel=True)(), test_impl1())

    def test_impl2(A, a):
        b = a
        return A[b]
    A = np.arange(10)
    a = slice(None)
    np.testing.assert_array_equal(njit(test_impl2, parallel=True)(A, a), test_impl2(A, a))