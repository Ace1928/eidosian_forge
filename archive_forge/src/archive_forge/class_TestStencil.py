import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
class TestStencil(TestStencilBase):

    def __init__(self, *args, **kwargs):
        super(TestStencil, self).__init__(*args, **kwargs)

    @skip_unsupported
    def test_stencil1(self):
        """Tests whether the optional out argument to stencil calls works.
        """

        def test_with_out(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.zeros(n ** 2).reshape((n, n))
            B = stencil1_kernel(A, out=B)
            return B

        def test_without_out(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = stencil1_kernel(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.zeros(n ** 2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    B[i, j] = 0.25 * (A[i, j + 1] + A[i + 1, j] + A[i, j - 1] + A[i - 1, j])
            return B
        n = 100
        self.check(test_impl_seq, test_with_out, n)
        self.check(test_impl_seq, test_without_out, n)

    @skip_unsupported
    def test_stencil2(self):
        """Tests whether the optional neighborhood argument to the stencil
        decorate works.
        """

        def test_seq(n):
            A = np.arange(n)
            B = stencil2_kernel(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(5, len(A)):
                B[i] = 0.3 * sum(A[i - 5:i + 1])
            return B
        n = 100
        self.check(test_impl_seq, test_seq, n)

        def test_seq(n, w):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                cum = a[-w]
                for i in range(-w + 1, w + 1):
                    cum += a[i]
                return 0.3 * cum
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w),))(A, w)
            return B

        def test_impl_seq(n, w):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(w, len(A) - w):
                B[i] = 0.3 * sum(A[i - w:i + w + 1])
            return B
        n = 100
        w = 5
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp))
        expected = test_impl_seq(n, w)
        parfor_output = cpfunc.entry_point(n, w)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

        def test_seq(n, w, offset):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                cum = a[-w + 1]
                for i in range(-w + 1, w + 1):
                    cum += a[i + 1]
                return 0.3 * cum
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w),), index_offsets=(-offset,))(A, w)
            return B
        offset = 1
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp, types.intp))
        parfor_output = cpfunc.entry_point(n, w, offset)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

        def test_seq(n, w, offset):
            A = np.arange(n)

            def stencil2_kernel(a, w):
                return 0.3 * np.sum(a[-w + 1:w + 2])
            B = numba.stencil(stencil2_kernel, neighborhood=((-w, w),), index_offsets=(-offset,))(A, w)
            return B
        offset = 1
        cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp, types.intp))
        parfor_output = cpfunc.entry_point(n, w, offset)
        np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
        self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

    @skip_unsupported
    def test_stencil3(self):
        """Tests whether a non-zero optional cval argument to the stencil
        decorator works.  Also tests integer result type.
        """

        def test_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = stencil3_kernel(A)
            return B
        test_njit = njit(test_seq)
        test_par = njit(test_seq, parallel=True)
        n = 5
        seq_res = test_seq(n)
        njit_res = test_njit(n)
        par_res = test_par(n)
        self.assertTrue(seq_res[0, 0] == 1.0 and seq_res[4, 4] == 1.0)
        self.assertTrue(njit_res[0, 0] == 1.0 and njit_res[4, 4] == 1.0)
        self.assertTrue(par_res[0, 0] == 1.0 and par_res[4, 4] == 1.0)

    @skip_unsupported
    def test_stencil_standard_indexing_1d(self):
        """Tests standard indexing with a 1d array.
        """

        def test_seq(n):
            A = np.arange(n)
            B = [3.0, 7.0]
            C = stencil_with_standard_indexing_1d(A, B)
            return C

        def test_impl_seq(n):
            A = np.arange(n)
            B = [3.0, 7.0]
            C = np.zeros(n)
            for i in range(1, n):
                C[i] = A[i - 1] * B[0] + A[i] * B[1]
            return C
        n = 100
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    def test_stencil_standard_indexing_2d(self):
        """Tests standard indexing with a 2d array and multiple stencil calls.
        """

        def test_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.ones((3, 3))
            C = stencil_with_standard_indexing_2d(A, B)
            D = stencil_with_standard_indexing_2d(C, B)
            return D

        def test_impl_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.ones((3, 3))
            C = np.zeros(n ** 2).reshape((n, n))
            D = np.zeros(n ** 2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    C[i, j] = A[i, j + 1] * B[0, 1] + A[i + 1, j] * B[1, 0] + A[i, j - 1] * B[0, -1] + A[i - 1, j] * B[-1, 0]
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    D[i, j] = C[i, j + 1] * B[0, 1] + C[i + 1, j] * B[1, 0] + C[i, j - 1] * B[0, -1] + C[i - 1, j] * B[-1, 0]
            return D
        n = 5
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    def test_stencil_multiple_inputs(self):
        """Tests whether multiple inputs of the same size work.
        """

        def test_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.arange(n ** 2).reshape((n, n))
            C = stencil_multiple_input_kernel(A, B)
            return C

        def test_impl_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.arange(n ** 2).reshape((n, n))
            C = np.zeros(n ** 2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    C[i, j] = 0.25 * (A[i, j + 1] + A[i + 1, j] + A[i, j - 1] + A[i - 1, j] + B[i, j + 1] + B[i + 1, j] + B[i, j - 1] + B[i - 1, j])
            return C
        n = 3
        self.check(test_impl_seq, test_seq, n)

        def test_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.arange(n ** 2).reshape((n, n))
            w = 0.25
            C = stencil_multiple_input_kernel_var(A, B, w)
            return C
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    def test_stencil_mixed_types(self):

        def test_impl_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = n ** 2 - np.arange(n ** 2).reshape((n, n))
            S = np.eye(n, dtype=np.bool_)
            O = np.zeros((n, n), dtype=A.dtype)
            for i in range(0, n):
                for j in range(0, n):
                    O[i, j] = A[i, j] if S[i, j] else B[i, j]
            return O

        def test_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = n ** 2 - np.arange(n ** 2).reshape((n, n))
            S = np.eye(n, dtype=np.bool_)
            O = stencil_multiple_input_mixed_types_2d(A, B, S)
            return O
        n = 3
        self.check(test_impl_seq, test_seq, n)

    @skip_unsupported
    def test_stencil_call(self):
        """Tests 2D numba.stencil calls.
        """

        def test_impl1(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.zeros(n ** 2).reshape((n, n))
            numba.stencil(lambda a: 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0]))(A, out=B)
            return B

        def test_impl2(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.zeros(n ** 2).reshape((n, n))

            def sf(a):
                return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])
            B = numba.stencil(sf)(A)
            return B

        def test_impl_seq(n):
            A = np.arange(n ** 2).reshape((n, n))
            B = np.zeros(n ** 2).reshape((n, n))
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    B[i, j] = 0.25 * (A[i, j + 1] + A[i + 1, j] + A[i, j - 1] + A[i - 1, j])
            return B
        n = 100
        self.check(test_impl_seq, test_impl1, n)
        self.check(test_impl_seq, test_impl2, n)

    @skip_unsupported
    def test_stencil_call_1D(self):
        """Tests 1D numba.stencil calls.
        """

        def test_impl(n):
            A = np.arange(n)
            B = np.zeros(n)
            numba.stencil(lambda a: 0.3 * (a[-1] + a[0] + a[1]))(A, out=B)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            for i in range(1, n - 1):
                B[i] = 0.3 * (A[i - 1] + A[i] + A[i + 1])
            return B
        n = 100
        self.check(test_impl_seq, test_impl, n)

    @skip_unsupported
    def test_stencil_call_const(self):
        """Tests numba.stencil call that has an index that can be inferred as
        constant from a unary expr. Otherwise, this would raise an error since
        neighborhood length is not specified.
        """

        def test_impl1(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 1
            numba.stencil(lambda a, c: 0.3 * (a[-c] + a[0] + a[c]))(A, c, out=B)
            return B

        def test_impl2(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 2
            numba.stencil(lambda a, c: 0.3 * (a[1 - c] + a[0] + a[c - 1]))(A, c, out=B)
            return B

        def test_impl3(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 2
            numba.stencil(lambda a, c: 0.3 * (a[-c + 1] + a[0] + a[c - 1]))(A, c, out=B)
            return B

        def test_impl4(n):
            A = np.arange(n)
            B = np.zeros(n)
            d = 1
            c = 2
            numba.stencil(lambda a, c, d: 0.3 * (a[-c + d] + a[0] + a[c - d]))(A, c, d, out=B)
            return B

        def test_impl_seq(n):
            A = np.arange(n)
            B = np.zeros(n)
            c = 1
            for i in range(1, n - 1):
                B[i] = 0.3 * (A[i - c] + A[i] + A[i + c])
            return B
        n = 100
        cpfunc1 = self.compile_parallel(test_impl1, (types.intp,))
        cpfunc2 = self.compile_parallel(test_impl2, (types.intp,))
        cpfunc3 = self.compile_parallel(test_impl3, (types.intp,))
        cpfunc4 = self.compile_parallel(test_impl4, (types.intp,))
        expected = test_impl_seq(n)
        parfor_output1 = cpfunc1.entry_point(n)
        parfor_output2 = cpfunc2.entry_point(n)
        parfor_output3 = cpfunc3.entry_point(n)
        parfor_output4 = cpfunc4.entry_point(n)
        np.testing.assert_almost_equal(parfor_output1, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output2, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output3, expected, decimal=3)
        np.testing.assert_almost_equal(parfor_output4, expected, decimal=3)
        with self.assertRaises(NumbaValueError) as e:
            test_impl4(4)
        self.assertIn("stencil kernel index is not constant, 'neighborhood' option required", str(e.exception))
        with self.assertRaises((LoweringError, NumbaValueError)) as e:
            njit(test_impl4)(4)
        self.assertIn("stencil kernel index is not constant, 'neighborhood' option required", str(e.exception))

    @skip_unsupported
    def test_stencil_parallel_off(self):
        """Tests 1D numba.stencil calls without parallel translation
           turned off.
        """

        def test_impl(A):
            return numba.stencil(lambda a: 0.3 * (a[-1] + a[0] + a[1]))(A)
        cpfunc = self.compile_parallel(test_impl, (numba.float64[:],), stencil=False)
        self.assertNotIn('@do_scheduling', cpfunc.library.get_llvm_str())

    @skip_unsupported
    def test_stencil_nested1(self):
        """Tests whether nested stencil decorator works.
        """

        @njit(parallel=True)
        def test_impl(n):

            @stencil
            def fun(a):
                c = 2
                return a[-c + 1]
            B = fun(n)
            return B

        def test_impl_seq(n):
            B = np.zeros(len(n), dtype=int)
            for i in range(1, len(n)):
                B[i] = n[i - 1]
            return B
        n = np.arange(10)
        np.testing.assert_equal(test_impl(n), test_impl_seq(n))

    @skip_unsupported
    def test_out_kwarg_w_cval(self):
        """ Issue #3518, out kwarg did not work with cval."""
        const_vals = [7, 7.0]

        def kernel(a):
            return a[0, 0] - a[1, 0]
        for const_val in const_vals:
            stencil_fn = numba.stencil(kernel, cval=const_val)

            def wrapped():
                A = np.arange(12).reshape((3, 4))
                ret = np.ones_like(A)
                stencil_fn(A, out=ret)
                return ret
            A = np.arange(12).reshape((3, 4))
            expected = np.full_like(A, -4)
            expected[-1, :] = const_val
            ret = np.ones_like(A)
            stencil_fn(A, out=ret)
            np.testing.assert_almost_equal(ret, expected)
            impls = self.compile_all(wrapped)
            for impl in impls:
                got = impl.entry_point()
                np.testing.assert_almost_equal(got, expected)
        stencil_fn = numba.stencil(kernel, cval=1j)

        def wrapped():
            A = np.arange(12).reshape((3, 4))
            ret = np.ones_like(A)
            stencil_fn(A, out=ret)
            return ret
        A = np.arange(12).reshape((3, 4))
        ret = np.ones_like(A)
        with self.assertRaises(NumbaValueError) as e:
            stencil_fn(A, out=ret)
        msg = 'cval type does not match stencil return type.'
        self.assertIn(msg, str(e.exception))
        for compiler in [self.compile_njit, self.compile_parallel]:
            try:
                compiler(wrapped, ())
            except (NumbaValueError, LoweringError) as e:
                self.assertIn(msg, str(e))
            else:
                raise AssertionError('Expected error was not raised')

    @skip_unsupported
    def test_out_kwarg_w_cval_np_attr(self):
        """ Test issue #7286 where the cval is a np attr/string-based numerical
        constant"""
        for cval in (np.nan, np.inf, -np.inf, float('inf'), -float('inf')):

            def kernel(a):
                return a[0, 0] - a[1, 0]
            stencil_fn = numba.stencil(kernel, cval=cval)

            def wrapped():
                A = np.arange(12.0).reshape((3, 4))
                ret = np.ones_like(A)
                stencil_fn(A, out=ret)
                return ret
            A = np.arange(12.0).reshape((3, 4))
            expected = np.full_like(A, -4)
            expected[-1, :] = cval
            ret = np.ones_like(A)
            stencil_fn(A, out=ret)
            np.testing.assert_almost_equal(ret, expected)
            impls = self.compile_all(wrapped)
            for impl in impls:
                got = impl.entry_point()
                np.testing.assert_almost_equal(got, expected)