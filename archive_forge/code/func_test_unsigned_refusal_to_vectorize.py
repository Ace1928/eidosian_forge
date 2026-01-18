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
@linux_only
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '0'})
def test_unsigned_refusal_to_vectorize(self):
    """ This checks that if fastmath is set and the underlying hardware
        is suitable, and the function supplied is amenable to fastmath based
        vectorization, that the vectorizer actually runs.
        """

    def will_not_vectorize(A):
        n = len(A)
        for i in range(-n, 0):
            A[i] = np.sqrt(A[i])
        return A

    def will_vectorize(A):
        n = len(A)
        for i in range(n):
            A[i] = np.sqrt(A[i])
        return A
    arg = np.zeros(10)
    self.assertFalse(config.BOUNDSCHECK)
    novec_asm = self.get_gufunc_asm(will_not_vectorize, 'signed', arg, fastmath=True)
    vec_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg, fastmath=True)
    for v in novec_asm.values():
        self.assertTrue('vsqrtpd' not in v)
        self.assertTrue('vsqrtsd' in v)
        self.assertTrue('zmm' not in v)
    for v in vec_asm.values():
        self.assertTrue('vsqrtpd' in v or '__svml_sqrt' in v)
        self.assertTrue('vmovupd' in v)
        self.assertTrue('zmm' in v)