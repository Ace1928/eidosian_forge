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
def test_transpose_3d(m, n, k):
    a = np.ones((m, n, k))
    b = a.T
    c = a.transpose()
    d = a.transpose(2, 0, 1)
    dt = a.transpose((2, 0, 1))
    e = a.transpose(0, 2, 1)
    et = a.transpose((0, 2, 1))