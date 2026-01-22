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
class TestPrangeSpecific(TestPrangeBase):
    """ Tests specific features/problems found under prange"""

    def test_prange_two_instances_same_reduction_var(self):

        def test_impl(n):
            c = 0
            for i in range(n):
                c += 1
                if i > 10:
                    c += 1
            return c
        self.prange_tester(test_impl, 9)

    def test_prange_conflicting_reduction_ops(self):

        def test_impl(n):
            c = 0
            for i in range(n):
                c += 1
                if i > 10:
                    c *= 1
            return c
        with self.assertRaises(errors.UnsupportedError) as raises:
            self.prange_tester(test_impl, 9)
        msg = 'Reduction variable c has multiple conflicting reduction operators.'
        self.assertIn(msg, str(raises.exception))

    def test_prange_two_conditional_reductions(self):

        def test_impl():
            A = B = 0
            for k in range(1):
                if k == 2:
                    A += 1
                else:
                    x = np.zeros((1, 1))
                    if x[0, 0]:
                        B += 1
            return (A, B)
        self.prange_tester(test_impl)

    def test_prange_nested_reduction1(self):

        def test_impl():
            A = 0
            for k in range(1):
                for i in range(1):
                    if i == 0:
                        A += 1
            return A
        self.prange_tester(test_impl)

    @disabled_test
    def test_check_error_model(self):

        def test_impl():
            n = 32
            A = np.zeros(n)
            for i in range(n):
                A[i] = 1 / i
            return A
        with self.assertRaises(ZeroDivisionError) as raises:
            test_impl()
        pfunc = self.generate_prange_func(test_impl, None)
        pcres = self.compile_parallel(pfunc, ())
        pfcres = self.compile_parallel_fastmath(pfunc, ())
        with self.assertRaises(ZeroDivisionError) as raises:
            pcres.entry_point()
        result = pfcres.entry_point()
        self.assertEqual(result[0], np.inf)

    def test_check_alias_analysis(self):

        def test_impl(A):
            for i in range(len(A)):
                B = A[i]
                B[:] = 1
            return A
        A = np.zeros(32).reshape(4, 8)
        self.prange_tester(test_impl, A, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)
        pfunc = self.generate_prange_func(test_impl, None)
        sig = tuple([numba.typeof(A)])
        cres = self.compile_parallel_fastmath(pfunc, sig)
        _ir = self._get_gufunc_ir(cres)
        for k, v in _ir.items():
            for line in v.splitlines():
                if 'define' in line and k in line:
                    self.assertEqual(line.count('noalias'), 2)
                    break

    def test_prange_raises_invalid_step_size(self):

        def test_impl(N):
            acc = 0
            for i in range(0, N, 2):
                acc += 2
            return acc
        with self.assertRaises(errors.UnsupportedRewriteError) as raises:
            self.prange_tester(test_impl, 1024)
        msg = 'Only constant step size of 1 is supported for prange'
        self.assertIn(msg, str(raises.exception))

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

    def test_parfor_alias1(self):

        def test_impl(n):
            b = np.zeros((n, n))
            a = b[0]
            for j in range(n):
                a[j] = j + 1
            return b.sum()
        self.prange_tester(test_impl, 4)

    def test_parfor_alias2(self):

        def test_impl(n):
            b = np.zeros((n, n))
            for i in range(n):
                a = b[i]
                for j in range(n):
                    a[j] = i + j
            return b.sum()
        self.prange_tester(test_impl, 4)

    def test_parfor_alias3(self):

        def test_impl(n):
            b = np.zeros((n, n, n))
            for i in range(n):
                a = b[i]
                for j in range(n):
                    c = a[j]
                    for k in range(n):
                        c[k] = i + j + k
            return b.sum()
        self.prange_tester(test_impl, 4)

    def test_parfor_race_1(self):

        def test_impl(x, y):
            for j in range(y):
                k = x
            return k
        raised_warnings = self.prange_tester(test_impl, 10, 20)
        warning_obj = raised_warnings[0]
        expected_msg = 'Variable k used in parallel loop may be written to simultaneously by multiple workers and may result in non-deterministic or unintended results.'
        self.assertIn(expected_msg, str(warning_obj.message))

    def test_nested_parfor_push_call_vars(self):
        """ issue 3686: if a prange has something inside it that causes
            a nested parfor to be generated and both the inner and outer
            parfor use the same call variable defined outside the parfors
            then ensure that when that call variable is pushed into the
            parfor that the call variable isn't duplicated with the same
            name resulting in a redundant type lock.
        """

        def test_impl():
            B = 0
            f = np.negative
            for i in range(1):
                this_matters = f(1.0)
                B += f(np.zeros(1))[0]
            for i in range(2):
                this_matters = f(1.0)
                B += f(np.zeros(1))[0]
            return B
        self.prange_tester(test_impl)

    def test_copy_global_for_parfor(self):
        """ issue4903: a global is copied next to a parfor so that
            it can be inlined into the parfor and thus not have to be
            passed to the parfor (i.e., an unsupported function type).
            This global needs to be renamed in the block into which
            it is copied.
        """

        def test_impl(zz, tc):
            lh = np.zeros(len(tc))
            lc = np.zeros(len(tc))
            for i in range(1):
                nt = tc[i]
                for t in range(nt):
                    lh += np.exp(zz[i, t])
                for t in range(nt):
                    lc += np.exp(zz[i, t])
            return (lh, lc)
        m = 2
        zz = np.ones((m, m, m))
        tc = np.ones(m, dtype=np.int_)
        self.prange_tester(test_impl, zz, tc, patch_instance=[0])

    def test_multiple_call_getattr_object(self):

        def test_impl(n):
            B = 0
            f = np.negative
            for i in range(1):
                this_matters = f(1.0)
                B += f(n)
            return B
        self.prange_tester(test_impl, 1.0)

    def test_argument_alias_recarray_field(self):

        def test_impl(n):
            for i in range(len(n)):
                n.x[i] = 7.0
            return n
        X1 = np.zeros(10, dtype=[('x', float), ('y', int)])
        X2 = np.zeros(10, dtype=[('x', float), ('y', int)])
        X3 = np.zeros(10, dtype=[('x', float), ('y', int)])
        v1 = X1.view(np.recarray)
        v2 = X2.view(np.recarray)
        v3 = X3.view(np.recarray)
        python_res = list(test_impl(v1))
        njit_res = list(njit(test_impl)(v2))
        pa_func = njit(test_impl, parallel=True)
        pa_res = list(pa_func(v3))
        self.assertEqual(python_res, njit_res)
        self.assertEqual(python_res, pa_res)

    def test_mutable_list_param(self):
        """ issue3699: test that mutable variable to call in loop
            is not hoisted.  The call in test_impl forces a manual
            check here rather than using prange_tester.
        """

        @njit
        def list_check(X):
            """ If the variable X is hoisted in the test_impl prange
                then subsequent list_check calls would return increasing
                values.
            """
            ret = X[-1]
            a = X[-1] + 1
            X.append(a)
            return ret

        def test_impl(n):
            for i in prange(n):
                X = [100]
                a = list_check(X)
            return a
        python_res = test_impl(10)
        njit_res = njit(test_impl)(10)
        pa_func = njit(test_impl, parallel=True)
        pa_res = pa_func(10)
        self.assertEqual(python_res, njit_res)
        self.assertEqual(python_res, pa_res)

    def test_list_comprehension_prange(self):

        def test_impl(x):
            return np.array([len(x[i]) for i in range(len(x))])
        x = [np.array([1, 2, 3], dtype=int), np.array([1, 2], dtype=int)]
        self.prange_tester(test_impl, x)

    def test_ssa_false_reduction(self):

        def test_impl(image, a, b):
            empty = np.zeros(image.shape)
            for i in range(image.shape[0]):
                r = image[i][0] / 255.0
                if a == 0:
                    h = 0
                if b == 0:
                    h = 0
                empty[i] = [h, h, h]
            return empty
        image = np.zeros((3, 3), dtype=np.int32)
        self.prange_tester(test_impl, image, 0, 0)

    def test_list_setitem_hoisting(self):

        def test_impl():
            n = 5
            a = np.empty(n, dtype=np.int64)
            for k in range(5):
                X = [0]
                X[0] = 1
                a[k] = X[0]
            return a
        self.prange_tester(test_impl)

    def test_record_array_setitem(self):
        state_dtype = np.dtype([('var', np.int32)])

        def test_impl(states):
            for i in range(1):
                states[i]['var'] = 1

        def comparer(a, b):
            assert a[0]['var'] == b[0]['var']
        self.prange_tester(test_impl, np.zeros(shape=1, dtype=state_dtype), check_arg_equality=[comparer])

    def test_record_array_setitem_yield_array(self):
        state_dtype = np.dtype([('x', np.intp)])

        def test_impl(states):
            n = states.size
            for i in range(states.size):
                states['x'][i] = 7 + i
            return states
        states = np.zeros(10, dtype=state_dtype)

        def comparer(a, b):
            np.testing.assert_equal(a, b)
        self.prange_tester(test_impl, states, check_arg_equality=[comparer])

    def test_issue7501(self):

        def test_impl(size, case):
            result = np.zeros((size,))
            if case == 1:
                for i in range(size):
                    result[i] += 1
            else:
                for i in range(size):
                    result[i] += 2
            return result[0]
        self.prange_tester(test_impl, 3, 1)

    def test_kde_example(self):

        def test_impl(X):
            b = 0.5
            points = np.array([-1.0, 2.0, 5.0])
            N = points.shape[0]
            n = X.shape[0]
            exps = 0
            for i in range(n):
                p = X[i]
                d = -(p - points) ** 2 / (2 * b ** 2)
                m = np.min(d)
                exps += m - np.log(b * N) + np.log(np.sum(np.exp(d - m)))
            return exps
        n = 128
        X = np.random.ranf(n)
        self.prange_tester(test_impl, X)

    @skip_parfors_unsupported
    def test_issue_due_to_max_label(self):
        out = subp.check_output([sys.executable, '-m', 'numba.tests.parfors_max_label_error'], timeout=30, stderr=subp.STDOUT)
        self.assertIn('TEST PASSED', out.decode())

    @skip_parfors_unsupported
    def test_issue7578(self):

        def test_impl(x):
            A = np.zeros_like(x)
            tmp = np.cos(x)
            for i in range(len(x)):
                A[i] = tmp.sum()
            return A
        x = np.arange(10.0)
        self.prange_tester(test_impl, x)