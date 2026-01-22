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
@skip_parfors_unsupported
class TestParforsDiagnostics(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def assert_fusion_equivalence(self, got, expected):
        a = self._fusion_equivalent(got)
        b = self._fusion_equivalent(expected)
        self.assertEqual(a, b)

    def _fusion_equivalent(self, thing):
        new = defaultdict(list)
        min_key = min(thing.keys())
        for k in sorted(thing.keys()):
            new[k - min_key] = [x - min_key for x in thing[k]]
        return new

    def assert_diagnostics(self, diagnostics, parfors_count=None, fusion_info=None, nested_fusion_info=None, replaced_fns=None, hoisted_allocations=None):
        if parfors_count is not None:
            self.assertEqual(parfors_count, diagnostics.count_parfors())
        if fusion_info is not None:
            self.assert_fusion_equivalence(fusion_info, diagnostics.fusion_info)
        if nested_fusion_info is not None:
            self.assert_fusion_equivalence(nested_fusion_info, diagnostics.nested_fusion_info)
        if replaced_fns is not None:
            repl = diagnostics.replaced_fns.values()
            for x in replaced_fns:
                for replaced in repl:
                    if replaced[0] == x:
                        break
                else:
                    msg = 'Replacement for %s was not found. Had %s' % (x, repl)
                    raise AssertionError(msg)
        if hoisted_allocations is not None:
            hoisted_allocs = diagnostics.hoisted_allocations()
            self.assertEqual(hoisted_allocations, len(hoisted_allocs))
        with captured_stdout():
            for x in range(1, 5):
                diagnostics.dump(x)

    def test_array_expr(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.zeros(n)
            return a + b
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1, fusion_info={3: [4, 5]})

    def test_prange(self):

        def test_impl():
            n = 10
            a = np.empty(n)
            for i in prange(n):
                a[i] = i * 10
            return a
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1)

    def test_user_varname(self):
        """make sure original user variable name is used in fusion info
        """

        def test_impl():
            n = 10
            x = np.ones(n)
            a = np.sin(x)
            b = np.cos(a * a)
            acc = 0
            for i in prange(n - 2):
                for j in prange(n - 1):
                    acc += b[i] + b[j + 1]
            return acc
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assertTrue(any(('slice(0, n, 1)' in r.message for r in diagnostics.fusion_reports)))

    def test_nested_prange(self):

        def test_impl():
            n = 10
            a = np.empty((n, n))
            for i in prange(n):
                for j in prange(n):
                    a[i, j] = i * 10 + j
            return a
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=2, nested_fusion_info={2: [1]})

    def test_function_replacement(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.argmin(a)
            return b
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1, fusion_info={2: [3]}, replaced_fns=[('argmin', 'numpy')])

    def test_reduction(self):

        def test_impl():
            n = 10
            a = np.ones(n + 1)
            acc = 0
            for i in prange(n):
                acc += a[i]
            return acc
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=2)

    def test_setitem(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            a[:] = 7
            return a
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1)

    def test_allocation_hoisting(self):

        def test_impl():
            n = 10
            m = 5
            acc = 0
            for i in prange(n):
                temp = np.zeros((m,))
                for j in range(m):
                    temp[j] = i
                acc += temp[-1]
            return acc
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, hoisted_allocations=1)