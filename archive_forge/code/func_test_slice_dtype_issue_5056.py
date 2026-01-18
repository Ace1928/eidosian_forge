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
def test_slice_dtype_issue_5056(self):

    @njit(parallel=True)
    def test_impl(data):
        N = data.shape[0]
        sums = np.zeros(N)
        for i in prange(N):
            sums[i] = np.sum(data[np.int32(0):np.int32(1)])
        return sums
    data = np.arange(10.0)
    np.testing.assert_array_equal(test_impl(data), test_impl.py_func(data))