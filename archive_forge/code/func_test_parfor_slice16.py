import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
@disabled_test
def test_parfor_slice16(self):
    """ This test is disabled because if n is larger than the array size
            then n and n-1 will both be the end of the array and thus the
            slices will in fact be of different sizes and unable to fuse.
        """

    def test_impl(a, b, n):
        assert a.shape == b.shape
        a[1:n] = 10
        b[0:n - 1] = 10
        return a * b
    self.check(test_impl, np.ones(10), np.zeros(10), 8)
    args = (numba.float64[:], numba.float64[:], numba.int64)
    self.assertEqual(countParfors(test_impl, args), 2)