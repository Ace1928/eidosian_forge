import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
class TestFBLAS2Simple:

    def test_gemv(self):
        for p in 'sd':
            f = getattr(fblas, p + 'gemv', None)
            if f is None:
                continue
            assert_array_almost_equal(f(3, [[3]], [-4]), [-36])
            assert_array_almost_equal(f(3, [[3]], [-4], 3, [5]), [-21])
        for p in 'cz':
            f = getattr(fblas, p + 'gemv', None)
            if f is None:
                continue
            assert_array_almost_equal(f(3j, [[3 - 4j]], [-4]), [-48 - 36j])
            assert_array_almost_equal(f(3j, [[3 - 4j]], [-4], 3, [5j]), [-48 - 21j])

    def test_ger(self):
        for p in 'sd':
            f = getattr(fblas, p + 'ger', None)
            if f is None:
                continue
            assert_array_almost_equal(f(1, [1, 2], [3, 4]), [[3, 4], [6, 8]])
            assert_array_almost_equal(f(2, [1, 2, 3], [3, 4]), [[6, 8], [12, 16], [18, 24]])
            assert_array_almost_equal(f(1, [1, 2], [3, 4], a=[[1, 2], [3, 4]]), [[4, 6], [9, 12]])
        for p in 'cz':
            f = getattr(fblas, p + 'geru', None)
            if f is None:
                continue
            assert_array_almost_equal(f(1, [1j, 2], [3, 4]), [[3j, 4j], [6, 8]])
            assert_array_almost_equal(f(-2, [1j, 2j, 3j], [3j, 4j]), [[6, 8], [12, 16], [18, 24]])
        for p in 'cz':
            for name in ('ger', 'gerc'):
                f = getattr(fblas, p + name, None)
                if f is None:
                    continue
                assert_array_almost_equal(f(1, [1j, 2], [3, 4]), [[3j, 4j], [6, 8]])
                assert_array_almost_equal(f(2, [1j, 2j, 3j], [3j, 4j]), [[6, 8], [12, 16], [18, 24]])

    def test_syr_her(self):
        x = np.arange(1, 5, dtype='d')
        resx = np.triu(x[:, np.newaxis] * x)
        resx_reverse = np.triu(x[::-1, np.newaxis] * x[::-1])
        y = np.linspace(0, 8.5, 17, endpoint=False)
        z = np.arange(1, 9, dtype='d').view('D')
        resz = np.triu(z[:, np.newaxis] * z)
        resz_reverse = np.triu(z[::-1, np.newaxis] * z[::-1])
        rehz = np.triu(z[:, np.newaxis] * z.conj())
        rehz_reverse = np.triu(z[::-1, np.newaxis] * z[::-1].conj())
        w = np.c_[np.zeros(4), z, np.zeros(4)].ravel()
        for p, rtol in zip('sd', [1e-07, 1e-14]):
            f = getattr(fblas, p + 'syr', None)
            if f is None:
                continue
            assert_allclose(f(1.0, x), resx, rtol=rtol)
            assert_allclose(f(1.0, x, lower=True), resx.T, rtol=rtol)
            assert_allclose(f(1.0, y, incx=2, offx=2, n=4), resx, rtol=rtol)
            assert_allclose(f(1.0, y, incx=-2, offx=2, n=4), resx_reverse, rtol=rtol)
            a = np.zeros((4, 4), 'f' if p == 's' else 'd', 'F')
            b = f(1.0, x, a=a, overwrite_a=True)
            assert_allclose(a, resx, rtol=rtol)
            b = f(2.0, x, a=a)
            assert_(a is not b)
            assert_allclose(b, 3 * resx, rtol=rtol)
            assert_raises(Exception, f, 1.0, x, incx=0)
            assert_raises(Exception, f, 1.0, x, offx=5)
            assert_raises(Exception, f, 1.0, x, offx=-2)
            assert_raises(Exception, f, 1.0, x, n=-2)
            assert_raises(Exception, f, 1.0, x, n=5)
            assert_raises(Exception, f, 1.0, x, lower=2)
            assert_raises(Exception, f, 1.0, x, a=np.zeros((2, 2), 'd', 'F'))
        for p, rtol in zip('cz', [1e-07, 1e-14]):
            f = getattr(fblas, p + 'syr', None)
            if f is None:
                continue
            assert_allclose(f(1.0, z), resz, rtol=rtol)
            assert_allclose(f(1.0, z, lower=True), resz.T, rtol=rtol)
            assert_allclose(f(1.0, w, incx=3, offx=1, n=4), resz, rtol=rtol)
            assert_allclose(f(1.0, w, incx=-3, offx=1, n=4), resz_reverse, rtol=rtol)
            a = np.zeros((4, 4), 'F' if p == 'c' else 'D', 'F')
            b = f(1.0, z, a=a, overwrite_a=True)
            assert_allclose(a, resz, rtol=rtol)
            b = f(2.0, z, a=a)
            assert_(a is not b)
            assert_allclose(b, 3 * resz, rtol=rtol)
            assert_raises(Exception, f, 1.0, x, incx=0)
            assert_raises(Exception, f, 1.0, x, offx=5)
            assert_raises(Exception, f, 1.0, x, offx=-2)
            assert_raises(Exception, f, 1.0, x, n=-2)
            assert_raises(Exception, f, 1.0, x, n=5)
            assert_raises(Exception, f, 1.0, x, lower=2)
            assert_raises(Exception, f, 1.0, x, a=np.zeros((2, 2), 'd', 'F'))
        for p, rtol in zip('cz', [1e-07, 1e-14]):
            f = getattr(fblas, p + 'her', None)
            if f is None:
                continue
            assert_allclose(f(1.0, z), rehz, rtol=rtol)
            assert_allclose(f(1.0, z, lower=True), rehz.T.conj(), rtol=rtol)
            assert_allclose(f(1.0, w, incx=3, offx=1, n=4), rehz, rtol=rtol)
            assert_allclose(f(1.0, w, incx=-3, offx=1, n=4), rehz_reverse, rtol=rtol)
            a = np.zeros((4, 4), 'F' if p == 'c' else 'D', 'F')
            b = f(1.0, z, a=a, overwrite_a=True)
            assert_allclose(a, rehz, rtol=rtol)
            b = f(2.0, z, a=a)
            assert_(a is not b)
            assert_allclose(b, 3 * rehz, rtol=rtol)
            assert_raises(Exception, f, 1.0, x, incx=0)
            assert_raises(Exception, f, 1.0, x, offx=5)
            assert_raises(Exception, f, 1.0, x, offx=-2)
            assert_raises(Exception, f, 1.0, x, n=-2)
            assert_raises(Exception, f, 1.0, x, n=5)
            assert_raises(Exception, f, 1.0, x, lower=2)
            assert_raises(Exception, f, 1.0, x, a=np.zeros((2, 2), 'd', 'F'))

    def test_syr2(self):
        x = np.arange(1, 5, dtype='d')
        y = np.arange(5, 9, dtype='d')
        resxy = np.triu(x[:, np.newaxis] * y + y[:, np.newaxis] * x)
        resxy_reverse = np.triu(x[::-1, np.newaxis] * y[::-1] + y[::-1, np.newaxis] * x[::-1])
        q = np.linspace(0, 8.5, 17, endpoint=False)
        for p, rtol in zip('sd', [1e-07, 1e-14]):
            f = getattr(fblas, p + 'syr2', None)
            if f is None:
                continue
            assert_allclose(f(1.0, x, y), resxy, rtol=rtol)
            assert_allclose(f(1.0, x, y, n=3), resxy[:3, :3], rtol=rtol)
            assert_allclose(f(1.0, x, y, lower=True), resxy.T, rtol=rtol)
            assert_allclose(f(1.0, q, q, incx=2, offx=2, incy=2, offy=10), resxy, rtol=rtol)
            assert_allclose(f(1.0, q, q, incx=2, offx=2, incy=2, offy=10, n=3), resxy[:3, :3], rtol=rtol)
            assert_allclose(f(1.0, q, q, incx=-2, offx=2, incy=-2, offy=10), resxy_reverse, rtol=rtol)
            a = np.zeros((4, 4), 'f' if p == 's' else 'd', 'F')
            b = f(1.0, x, y, a=a, overwrite_a=True)
            assert_allclose(a, resxy, rtol=rtol)
            b = f(2.0, x, y, a=a)
            assert_(a is not b)
            assert_allclose(b, 3 * resxy, rtol=rtol)
            assert_raises(Exception, f, 1.0, x, y, incx=0)
            assert_raises(Exception, f, 1.0, x, y, offx=5)
            assert_raises(Exception, f, 1.0, x, y, offx=-2)
            assert_raises(Exception, f, 1.0, x, y, incy=0)
            assert_raises(Exception, f, 1.0, x, y, offy=5)
            assert_raises(Exception, f, 1.0, x, y, offy=-2)
            assert_raises(Exception, f, 1.0, x, y, n=-2)
            assert_raises(Exception, f, 1.0, x, y, n=5)
            assert_raises(Exception, f, 1.0, x, y, lower=2)
            assert_raises(Exception, f, 1.0, x, y, a=np.zeros((2, 2), 'd', 'F'))

    def test_her2(self):
        x = np.arange(1, 9, dtype='d').view('D')
        y = np.arange(9, 17, dtype='d').view('D')
        resxy = x[:, np.newaxis] * y.conj() + y[:, np.newaxis] * x.conj()
        resxy = np.triu(resxy)
        resxy_reverse = x[::-1, np.newaxis] * y[::-1].conj()
        resxy_reverse += y[::-1, np.newaxis] * x[::-1].conj()
        resxy_reverse = np.triu(resxy_reverse)
        u = np.c_[np.zeros(4), x, np.zeros(4)].ravel()
        v = np.c_[np.zeros(4), y, np.zeros(4)].ravel()
        for p, rtol in zip('cz', [1e-07, 1e-14]):
            f = getattr(fblas, p + 'her2', None)
            if f is None:
                continue
            assert_allclose(f(1.0, x, y), resxy, rtol=rtol)
            assert_allclose(f(1.0, x, y, n=3), resxy[:3, :3], rtol=rtol)
            assert_allclose(f(1.0, x, y, lower=True), resxy.T.conj(), rtol=rtol)
            assert_allclose(f(1.0, u, v, incx=3, offx=1, incy=3, offy=1), resxy, rtol=rtol)
            assert_allclose(f(1.0, u, v, incx=3, offx=1, incy=3, offy=1, n=3), resxy[:3, :3], rtol=rtol)
            assert_allclose(f(1.0, u, v, incx=-3, offx=1, incy=-3, offy=1), resxy_reverse, rtol=rtol)
            a = np.zeros((4, 4), 'F' if p == 'c' else 'D', 'F')
            b = f(1.0, x, y, a=a, overwrite_a=True)
            assert_allclose(a, resxy, rtol=rtol)
            b = f(2.0, x, y, a=a)
            assert_(a is not b)
            assert_allclose(b, 3 * resxy, rtol=rtol)
            assert_raises(Exception, f, 1.0, x, y, incx=0)
            assert_raises(Exception, f, 1.0, x, y, offx=5)
            assert_raises(Exception, f, 1.0, x, y, offx=-2)
            assert_raises(Exception, f, 1.0, x, y, incy=0)
            assert_raises(Exception, f, 1.0, x, y, offy=5)
            assert_raises(Exception, f, 1.0, x, y, offy=-2)
            assert_raises(Exception, f, 1.0, x, y, n=-2)
            assert_raises(Exception, f, 1.0, x, y, n=5)
            assert_raises(Exception, f, 1.0, x, y, lower=2)
            assert_raises(Exception, f, 1.0, x, y, a=np.zeros((2, 2), 'd', 'F'))

    def test_gbmv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 7
            m = 5
            kl = 1
            ku = 2
            A = toeplitz(append(rand(kl + 1), zeros(m - kl - 1)), append(rand(ku + 1), zeros(n - ku - 1)))
            A = A.astype(dtype)
            Ab = zeros((kl + ku + 1, n), dtype=dtype)
            Ab[2, :5] = A[0, 0]
            Ab[1, 1:6] = A[0, 1]
            Ab[0, 2:7] = A[0, 2]
            Ab[3, :4] = A[1, 0]
            x = rand(n).astype(dtype)
            y = rand(m).astype(dtype)
            alpha, beta = (dtype(3), dtype(-5))
            func, = get_blas_funcs(('gbmv',), dtype=dtype)
            y1 = func(m=m, n=n, ku=ku, kl=kl, alpha=alpha, a=Ab, x=x, y=y, beta=beta)
            y2 = alpha * A.dot(x) + beta * y
            assert_array_almost_equal(y1, y2)
            y1 = func(m=m, n=n, ku=ku, kl=kl, alpha=alpha, a=Ab, x=y, y=x, beta=beta, trans=1)
            y2 = alpha * A.T.dot(y) + beta * x
            assert_array_almost_equal(y1, y2)

    def test_sbmv_hbmv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 6
            k = 2
            A = zeros((n, n), dtype=dtype)
            Ab = zeros((k + 1, n), dtype=dtype)
            A[arange(n), arange(n)] = rand(n)
            for ind2 in range(1, k + 1):
                temp = rand(n - ind2)
                A[arange(n - ind2), arange(ind2, n)] = temp
                Ab[-1 - ind2, ind2:] = temp
            A = A.astype(dtype)
            A = A + A.T if ind < 2 else A + A.conj().T
            Ab[-1, :] = diag(A)
            x = rand(n).astype(dtype)
            y = rand(n).astype(dtype)
            alpha, beta = (dtype(1.25), dtype(3))
            if ind > 1:
                func, = get_blas_funcs(('hbmv',), dtype=dtype)
            else:
                func, = get_blas_funcs(('sbmv',), dtype=dtype)
            y1 = func(k=k, alpha=alpha, a=Ab, x=x, y=y, beta=beta)
            y2 = alpha * A.dot(x) + beta * y
            assert_array_almost_equal(y1, y2)

    def test_spmv_hpmv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES + COMPLEX_DTYPES):
            n = 3
            A = rand(n, n).astype(dtype)
            if ind > 1:
                A += rand(n, n) * 1j
            A = A.astype(dtype)
            A = A + A.T if ind < 4 else A + A.conj().T
            c, r = tril_indices(n)
            Ap = A[r, c]
            x = rand(n).astype(dtype)
            y = rand(n).astype(dtype)
            xlong = arange(2 * n).astype(dtype)
            ylong = ones(2 * n).astype(dtype)
            alpha, beta = (dtype(1.25), dtype(2))
            if ind > 3:
                func, = get_blas_funcs(('hpmv',), dtype=dtype)
            else:
                func, = get_blas_funcs(('spmv',), dtype=dtype)
            y1 = func(n=n, alpha=alpha, ap=Ap, x=x, y=y, beta=beta)
            y2 = alpha * A.dot(x) + beta * y
            assert_array_almost_equal(y1, y2)
            y1 = func(n=n - 1, alpha=alpha, beta=beta, x=xlong, y=ylong, ap=Ap, incx=2, incy=2, offx=n, offy=n)
            y2 = (alpha * A[:-1, :-1]).dot(xlong[3::2]) + beta * ylong[3::2]
            assert_array_almost_equal(y1[3::2], y2)
            assert_almost_equal(y1[4], ylong[4])

    def test_spr_hpr(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES + COMPLEX_DTYPES):
            n = 3
            A = rand(n, n).astype(dtype)
            if ind > 1:
                A += rand(n, n) * 1j
            A = A.astype(dtype)
            A = A + A.T if ind < 4 else A + A.conj().T
            c, r = tril_indices(n)
            Ap = A[r, c]
            x = rand(n).astype(dtype)
            alpha = (DTYPES + COMPLEX_DTYPES)[mod(ind, 4)](2.5)
            if ind > 3:
                func, = get_blas_funcs(('hpr',), dtype=dtype)
                y2 = alpha * x[:, None].dot(x[None, :].conj()) + A
            else:
                func, = get_blas_funcs(('spr',), dtype=dtype)
                y2 = alpha * x[:, None].dot(x[None, :]) + A
            y1 = func(n=n, alpha=alpha, ap=Ap, x=x)
            y1f = zeros((3, 3), dtype=dtype)
            y1f[r, c] = y1
            y1f[c, r] = y1.conj() if ind > 3 else y1
            assert_array_almost_equal(y1f, y2)

    def test_spr2_hpr2(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 3
            A = rand(n, n).astype(dtype)
            if ind > 1:
                A += rand(n, n) * 1j
            A = A.astype(dtype)
            A = A + A.T if ind < 2 else A + A.conj().T
            c, r = tril_indices(n)
            Ap = A[r, c]
            x = rand(n).astype(dtype)
            y = rand(n).astype(dtype)
            alpha = dtype(2)
            if ind > 1:
                func, = get_blas_funcs(('hpr2',), dtype=dtype)
            else:
                func, = get_blas_funcs(('spr2',), dtype=dtype)
            u = alpha.conj() * x[:, None].dot(y[None, :].conj())
            y2 = A + u + u.conj().T
            y1 = func(n=n, alpha=alpha, x=x, y=y, ap=Ap)
            y1f = zeros((3, 3), dtype=dtype)
            y1f[r, c] = y1
            y1f[[1, 2, 2], [0, 0, 1]] = y1[[1, 3, 4]].conj()
            assert_array_almost_equal(y1f, y2)

    def test_tbmv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 10
            k = 3
            x = rand(n).astype(dtype)
            A = zeros((n, n), dtype=dtype)
            for sup in range(k + 1):
                A[arange(n - sup), arange(sup, n)] = rand(n - sup)
            if ind > 1:
                A[nonzero(A)] += 1j * rand((k + 1) * n - k * (k + 1) // 2).astype(dtype)
            Ab = zeros((k + 1, n), dtype=dtype)
            for row in range(k + 1):
                Ab[-row - 1, row:] = diag(A, k=row)
            func, = get_blas_funcs(('tbmv',), dtype=dtype)
            y1 = func(k=k, a=Ab, x=x)
            y2 = A.dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(k=k, a=Ab, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = A.dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(k=k, a=Ab, x=x, diag=1, trans=1)
            y2 = A.T.dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(k=k, a=Ab, x=x, diag=1, trans=2)
            y2 = A.conj().T.dot(x)
            assert_array_almost_equal(y1, y2)

    def test_tbsv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 6
            k = 3
            x = rand(n).astype(dtype)
            A = zeros((n, n), dtype=dtype)
            for sup in range(k + 1):
                A[arange(n - sup), arange(sup, n)] = rand(n - sup)
            if ind > 1:
                A[nonzero(A)] += 1j * rand((k + 1) * n - k * (k + 1) // 2).astype(dtype)
            Ab = zeros((k + 1, n), dtype=dtype)
            for row in range(k + 1):
                Ab[-row - 1, row:] = diag(A, k=row)
            func, = get_blas_funcs(('tbsv',), dtype=dtype)
            y1 = func(k=k, a=Ab, x=x)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)
            y1 = func(k=k, a=Ab, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)
            y1 = func(k=k, a=Ab, x=x, diag=1, trans=1)
            y2 = solve(A.T, x)
            assert_array_almost_equal(y1, y2)
            y1 = func(k=k, a=Ab, x=x, diag=1, trans=2)
            y2 = solve(A.conj().T, x)
            assert_array_almost_equal(y1, y2)

    def test_tpmv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 10
            x = rand(n).astype(dtype)
            A = triu(rand(n, n)) if ind < 2 else triu(rand(n, n) + rand(n, n) * 1j)
            c, r = tril_indices(n)
            Ap = A[r, c]
            func, = get_blas_funcs(('tpmv',), dtype=dtype)
            y1 = func(n=n, ap=Ap, x=x)
            y2 = A.dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(n=n, ap=Ap, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = A.dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=1)
            y2 = A.T.dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=2)
            y2 = A.conj().T.dot(x)
            assert_array_almost_equal(y1, y2)

    def test_tpsv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 10
            x = rand(n).astype(dtype)
            A = triu(rand(n, n)) if ind < 2 else triu(rand(n, n) + rand(n, n) * 1j)
            A += eye(n)
            c, r = tril_indices(n)
            Ap = A[r, c]
            func, = get_blas_funcs(('tpsv',), dtype=dtype)
            y1 = func(n=n, ap=Ap, x=x)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)
            y1 = func(n=n, ap=Ap, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)
            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=1)
            y2 = solve(A.T, x)
            assert_array_almost_equal(y1, y2)
            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=2)
            y2 = solve(A.conj().T, x)
            assert_array_almost_equal(y1, y2)

    def test_trmv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 3
            A = (rand(n, n) + eye(n)).astype(dtype)
            x = rand(3).astype(dtype)
            func, = get_blas_funcs(('trmv',), dtype=dtype)
            y1 = func(a=A, x=x)
            y2 = triu(A).dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(a=A, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = triu(A).dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(a=A, x=x, diag=1, trans=1)
            y2 = triu(A).T.dot(x)
            assert_array_almost_equal(y1, y2)
            y1 = func(a=A, x=x, diag=1, trans=2)
            y2 = triu(A).conj().T.dot(x)
            assert_array_almost_equal(y1, y2)

    def test_trsv(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 15
            A = (rand(n, n) + eye(n)).astype(dtype)
            x = rand(n).astype(dtype)
            func, = get_blas_funcs(('trsv',), dtype=dtype)
            y1 = func(a=A, x=x)
            y2 = solve(triu(A), x)
            assert_array_almost_equal(y1, y2)
            y1 = func(a=A, x=x, lower=1)
            y2 = solve(tril(A), x)
            assert_array_almost_equal(y1, y2)
            y1 = func(a=A, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = solve(triu(A), x)
            assert_array_almost_equal(y1, y2)
            y1 = func(a=A, x=x, diag=1, trans=1)
            y2 = solve(triu(A).T, x)
            assert_array_almost_equal(y1, y2)
            y1 = func(a=A, x=x, diag=1, trans=2)
            y2 = solve(triu(A).conj().T, x)
            assert_array_almost_equal(y1, y2)