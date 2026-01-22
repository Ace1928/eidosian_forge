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
class TestPrangeBasic(TestPrangeBase):
    """ Tests Prange """

    def test_prange01(self):

        def test_impl():
            n = 4
            A = np.zeros(n)
            for i in range(n):
                A[i] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange02(self):

        def test_impl():
            n = 4
            A = np.zeros(n - 1)
            for i in range(1, n):
                A[i - 1] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03(self):

        def test_impl():
            s = 10
            for i in range(10):
                s += 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03mul(self):

        def test_impl():
            s = 3
            for i in range(10):
                s *= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03sub(self):

        def test_impl():
            s = 100
            for i in range(10):
                s -= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03div(self):

        def test_impl():
            s = 10
            for i in range(10):
                s /= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange04(self):

        def test_impl():
            a = 2
            b = 3
            A = np.empty(4)
            for i in range(4):
                if i == a:
                    A[i] = b
                else:
                    A[i] = 0
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange05(self):

        def test_impl():
            n = 4
            A = np.ones(n, dtype=np.float64)
            s = 0
            for i in range(1, n - 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange06(self):

        def test_impl():
            n = 4
            A = np.ones(n, dtype=np.float64)
            s = 0
            for i in range(1, 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange07(self):

        def test_impl():
            n = 4
            A = np.ones(n, dtype=np.float64)
            s = 0
            for i in range(n, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange08(self):

        def test_impl():
            n = 4
            A = np.ones(n)
            acc = 0
            for i in range(len(A)):
                for j in range(len(A)):
                    acc += A[i]
            return acc
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange08_1(self):

        def test_impl():
            n = 4
            A = np.ones(n)
            acc = 0
            for i in range(4):
                for j in range(4):
                    acc += A[i]
            return acc
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange09(self):

        def test_impl():
            n = 4
            acc = 0
            for i in range(n):
                for j in range(n):
                    acc += 1
            return acc
        self.prange_tester(test_impl, patch_instance=[1], scheduler_type='unsigned', check_fastmath=True)

    def test_prange10(self):

        def test_impl():
            n = 4
            acc2 = 0
            for j in range(n):
                acc1 = 0
                for i in range(n):
                    acc1 += 1
                acc2 += acc1
            return acc2
        self.prange_tester(test_impl, patch_instance=[0], scheduler_type='unsigned', check_fastmath=True)

    @unittest.skip('list append is not thread-safe yet (#2391, #2408)')
    def test_prange11(self):

        def test_impl():
            n = 4
            return [np.sin(j) for j in range(n)]
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange12(self):

        def test_impl():
            acc = 0
            n = 4
            X = np.ones(n)
            for i in range(-len(X)):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange13(self):

        def test_impl(n):
            acc = 0
            for i in range(n):
                acc += 1
            return acc
        self.prange_tester(test_impl, np.int32(4), scheduler_type='unsigned', check_fastmath=True)

    def test_prange14(self):

        def test_impl(A):
            s = 3
            for i in range(len(A)):
                s += A[i] * 2
            return s
        self.prange_tester(test_impl, np.random.ranf(4), scheduler_type='unsigned', check_fastmath=True)

    def test_prange15(self):

        def test_impl(N):
            acc = 0
            for i in range(N):
                x = np.ones((1, 1))
                acc += x[0, 0]
            return acc
        self.prange_tester(test_impl, 1024, scheduler_type='unsigned', check_fastmath=True)

    def test_prange16(self):

        def test_impl(N):
            acc = 0
            for i in range(-N, N):
                acc += 2
            return acc
        self.prange_tester(test_impl, 1024, scheduler_type='signed', check_fastmath=True)

    def test_prange17(self):

        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-N, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange18(self):

        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-N, 5):
                acc += X[i]
                for j in range(-4, N):
                    acc += X[j]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange19(self):

        def test_impl(N):
            acc = 0
            M = N + 4
            X = np.ones((N, M))
            for i in range(-N, N):
                for j in range(-M, M):
                    acc += X[i, j]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange20(self):

        def test_impl(N):
            acc = 0
            X = np.ones(N)
            for i in range(-1, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange21(self):

        def test_impl(N):
            acc = 0
            for i in range(-3, -1):
                acc += 3
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange22(self):

        def test_impl():
            a = 0
            b = 3
            A = np.empty(4)
            for i in range(-2, 2):
                if i == a:
                    A[i] = b
                elif i < 1:
                    A[i] = -1
                else:
                    A[i] = 7
            return A
        self.prange_tester(test_impl, scheduler_type='signed', check_fastmath=True, check_fastmath_result=True)

    def test_prange23(self):

        def test_impl(A):
            for i in range(len(A)):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)

    def test_prange24(self):

        def test_impl(A):
            for i in range(-len(A), 0):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='signed', check_fastmath=True, check_fastmath_result=True)

    def test_prange25(self):

        def test_impl(A):
            n = len(A)
            buf = [np.zeros_like(A) for _ in range(n)]
            for i in range(n):
                buf[i] = A + i
            return buf
        A = np.ones((10,))
        self.prange_tester(test_impl, A, patch_instance=[1], scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(A),))
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        hoisted_allocs = diagnostics.hoisted_allocations()
        self.assertEqual(len(hoisted_allocs), 0)

    def test_prange26(self):

        def test_impl(A):
            B = A[::3]
            for i in range(len(B)):
                B[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)

    def test_prange27(self):

        def test_impl(a, b, c):
            for j in range(b[0] - 1):
                for k in range(2):
                    z = np.abs(a[c - 1:c + 1])
            return 0
        self.prange_tester(test_impl, np.arange(20), np.asarray([4, 4, 4, 4, 4, 4, 4, 4, 4, 4]), 0, patch_instance=[1], scheduler_type='unsigned', check_fastmath=True)

    def test_prange28(self):

        def test_impl(x, y):
            out = np.zeros(len(y))
            for idx in range(0, len(y)):
                i0 = y[idx, 0]
                i1 = y[idx, 1]
                Pt1 = x[i0]
                Pt2 = x[i1]
                v = Pt1 - Pt2
                vl2 = v[0] + v[1]
                out[idx] = vl2
            return out
        X = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
        Y = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        self.prange_tester(test_impl, X, Y, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)

    def test_prange29(self):

        def test_impl(flag):
            result = 0
            if flag:
                for i in range(1):
                    result += 1
            else:
                for i in range(1):
                    result -= 3
            return result
        self.prange_tester(test_impl, True)
        self.prange_tester(test_impl, False)

    def test_prange30(self):

        def test_impl(x, par, numthreads):
            n_par = par.shape[0]
            n_x = len(x)
            result = np.zeros((n_par, n_x), dtype=np.float64)
            chunklen = (len(x) + numthreads - 1) // numthreads
            for i in range(numthreads):
                start = i * chunklen
                stop = (i + 1) * chunklen
                result[:, start:stop] = x[start:stop] * par[:]
            return result
        x = np.array(np.arange(0, 6, 1.0))
        par = np.array([1.0, 2.0, 3.0])
        self.prange_tester(test_impl, x, par, 2)