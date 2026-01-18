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
def test_prange_fastmath_check_works(self):

    def test_impl():
        n = 128
        A = 0
        for i in range(n):
            A += i / 2.0
        return A
    self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)
    pfunc = self.generate_prange_func(test_impl, None)
    cres = self.compile_parallel_fastmath(pfunc, ())
    ir = self._get_gufunc_ir(cres)
    _id = '%[A-Z_0-9]?(.[0-9]+)+[.]?[i]?'
    recipr_str = '\\s+%s = fmul fast double %s, 5.000000e-01'
    reciprocal_inst = re.compile(recipr_str % (_id, _id))
    fadd_inst = re.compile('\\s+%s = fadd fast double %s, %s' % (_id, _id, _id))
    found = False
    for name, kernel in ir.items():
        if name in cres.library.get_llvm_str():
            splitted = kernel.splitlines()
            for i, x in enumerate(splitted):
                if reciprocal_inst.match(x):
                    self.assertTrue(fadd_inst.match(splitted[i + 1]))
                    found = True
                    break
    self.assertTrue(found, 'fast instruction pattern was not found.')