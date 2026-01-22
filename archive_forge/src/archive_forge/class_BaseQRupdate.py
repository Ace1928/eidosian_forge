import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
class BaseQRupdate(BaseQRdeltas):

    def generate(self, type, mode='full', p=1):
        a, q, r = super().generate(type, mode)
        if p == 1:
            u = np.random.random(q.shape[0])
            v = np.random.random(r.shape[1])
        else:
            u = np.random.random((q.shape[0], p))
            v = np.random.random((r.shape[1], p))
        if np.iscomplexobj(self.dtype.type(1)):
            b = np.random.random(u.shape)
            u = u + 1j * b
            c = np.random.random(v.shape)
            v = v + 1j * c
        u = u.astype(self.dtype)
        v = v.astype(self.dtype)
        return (a, q, r, u, v)

    def test_sqr_rank_1(self):
        a, q, r, u, v = self.generate('sqr')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_sqr_rank_p(self):
        for p in [1, 2, 3, 5]:
            a, q, r, u, v = self.generate('sqr', p=p)
            if p == 1:
                u = u.reshape(u.size, 1)
                v = v.reshape(v.size, 1)
            q1, r1 = qr_update(q, r, u, v, False)
            a1 = a + np.dot(u, v.T.conj())
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_tall_rank_1(self):
        a, q, r, u, v = self.generate('tall')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_tall_rank_p(self):
        for p in [1, 2, 3, 5]:
            a, q, r, u, v = self.generate('tall', p=p)
            if p == 1:
                u = u.reshape(u.size, 1)
                v = v.reshape(v.size, 1)
            q1, r1 = qr_update(q, r, u, v, False)
            a1 = a + np.dot(u, v.T.conj())
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_fat_rank_1(self):
        a, q, r, u, v = self.generate('fat')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_fat_rank_p(self):
        for p in [1, 2, 3, 5]:
            a, q, r, u, v = self.generate('fat', p=p)
            if p == 1:
                u = u.reshape(u.size, 1)
                v = v.reshape(v.size, 1)
            q1, r1 = qr_update(q, r, u, v, False)
            a1 = a + np.dot(u, v.T.conj())
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_economic_rank_1(self):
        a, q, r, u, v = self.generate('tall', 'economic')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_economic_rank_p(self):
        for p in [1, 2, 3, 5]:
            a, q, r, u, v = self.generate('tall', 'economic', p)
            if p == 1:
                u = u.reshape(u.size, 1)
                v = v.reshape(v.size, 1)
            q1, r1 = qr_update(q, r, u, v, False)
            a1 = a + np.dot(u, v.T.conj())
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_Mx1_rank_1(self):
        a, q, r, u, v = self.generate('Mx1')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_Mx1_rank_p(self):
        a, q, r, u, v = self.generate('Mx1', p=1)
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.dot(u, v.T.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_Mx1_economic_rank_1(self):
        a, q, r, u, v = self.generate('Mx1', 'economic')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_Mx1_economic_rank_p(self):
        a, q, r, u, v = self.generate('Mx1', 'economic', p=1)
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.dot(u, v.T.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_1xN_rank_1(self):
        a, q, r, u, v = self.generate('1xN')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1xN_rank_p(self):
        a, q, r, u, v = self.generate('1xN', p=1)
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.dot(u, v.T.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_rank_1(self):
        a, q, r, u, v = self.generate('1x1')
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_rank_p(self):
        a, q, r, u, v = self.generate('1x1', p=1)
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        q1, r1 = qr_update(q, r, u, v, False)
        a1 = a + np.dot(u, v.T.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_rank_1_scalar(self):
        a, q, r, u, v = self.generate('1x1')
        assert_raises(ValueError, qr_update, q[0, 0], r, u, v)
        assert_raises(ValueError, qr_update, q, r[0, 0], u, v)
        assert_raises(ValueError, qr_update, q, r, u[0], v)
        assert_raises(ValueError, qr_update, q, r, u, v[0])

    def base_non_simple_strides(self, adjust_strides, mode, p, overwriteable):
        assert_sqr = False if mode == 'economic' else True
        for type in ['sqr', 'tall', 'fat']:
            a, q0, r0, u0, v0 = self.generate(type, mode, p)
            qs, rs, us, vs = adjust_strides((q0, r0, u0, v0))
            if p == 1:
                aup = a + np.outer(u0, v0.conj())
            else:
                aup = a + np.dot(u0, v0.T.conj())
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            v = v0.copy('C')
            q1, r1 = qr_update(qs, r, u, v, False)
            check_qr(q1, r1, aup, self.rtol, self.atol, assert_sqr)
            q1o, r1o = qr_update(qs, r, u, v, True)
            check_qr(q1o, r1o, aup, self.rtol, self.atol, assert_sqr)
            if overwriteable:
                assert_allclose(r1o, r, rtol=self.rtol, atol=self.atol)
                assert_allclose(v, v0.conj(), rtol=self.rtol, atol=self.atol)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            v = v0.copy('C')
            q2, r2 = qr_update(q, rs, u, v, False)
            check_qr(q2, r2, aup, self.rtol, self.atol, assert_sqr)
            q2o, r2o = qr_update(q, rs, u, v, True)
            check_qr(q2o, r2o, aup, self.rtol, self.atol, assert_sqr)
            if overwriteable:
                assert_allclose(r2o, rs, rtol=self.rtol, atol=self.atol)
                assert_allclose(v, v0.conj(), rtol=self.rtol, atol=self.atol)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            v = v0.copy('C')
            q3, r3 = qr_update(q, r, us, v, False)
            check_qr(q3, r3, aup, self.rtol, self.atol, assert_sqr)
            q3o, r3o = qr_update(q, r, us, v, True)
            check_qr(q3o, r3o, aup, self.rtol, self.atol, assert_sqr)
            if overwriteable:
                assert_allclose(r3o, r, rtol=self.rtol, atol=self.atol)
                assert_allclose(v, v0.conj(), rtol=self.rtol, atol=self.atol)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            v = v0.copy('C')
            q4, r4 = qr_update(q, r, u, vs, False)
            check_qr(q4, r4, aup, self.rtol, self.atol, assert_sqr)
            q4o, r4o = qr_update(q, r, u, vs, True)
            check_qr(q4o, r4o, aup, self.rtol, self.atol, assert_sqr)
            if overwriteable:
                assert_allclose(r4o, r, rtol=self.rtol, atol=self.atol)
                assert_allclose(vs, v0.conj(), rtol=self.rtol, atol=self.atol)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            v = v0.copy('C')
            qs, rs, us, vs = adjust_strides((q, r, u, v))
            q5, r5 = qr_update(qs, rs, us, vs, False)
            check_qr(q5, r5, aup, self.rtol, self.atol, assert_sqr)
            q5o, r5o = qr_update(qs, rs, us, vs, True)
            check_qr(q5o, r5o, aup, self.rtol, self.atol, assert_sqr)
            if overwriteable:
                assert_allclose(r5o, rs, rtol=self.rtol, atol=self.atol)
                assert_allclose(vs, v0.conj(), rtol=self.rtol, atol=self.atol)

    def test_non_unit_strides_rank_1(self):
        self.base_non_simple_strides(make_strided, 'full', 1, True)

    def test_non_unit_strides_economic_rank_1(self):
        self.base_non_simple_strides(make_strided, 'economic', 1, True)

    def test_non_unit_strides_rank_p(self):
        self.base_non_simple_strides(make_strided, 'full', 3, False)

    def test_non_unit_strides_economic_rank_p(self):
        self.base_non_simple_strides(make_strided, 'economic', 3, False)

    def test_neg_strides_rank_1(self):
        self.base_non_simple_strides(negate_strides, 'full', 1, False)

    def test_neg_strides_economic_rank_1(self):
        self.base_non_simple_strides(negate_strides, 'economic', 1, False)

    def test_neg_strides_rank_p(self):
        self.base_non_simple_strides(negate_strides, 'full', 3, False)

    def test_neg_strides_economic_rank_p(self):
        self.base_non_simple_strides(negate_strides, 'economic', 3, False)

    def test_non_itemsize_strides_rank_1(self):
        self.base_non_simple_strides(nonitemsize_strides, 'full', 1, False)

    def test_non_itemsize_strides_economic_rank_1(self):
        self.base_non_simple_strides(nonitemsize_strides, 'economic', 1, False)

    def test_non_itemsize_strides_rank_p(self):
        self.base_non_simple_strides(nonitemsize_strides, 'full', 3, False)

    def test_non_itemsize_strides_economic_rank_p(self):
        self.base_non_simple_strides(nonitemsize_strides, 'economic', 3, False)

    def test_non_native_byte_order_rank_1(self):
        self.base_non_simple_strides(make_nonnative, 'full', 1, False)

    def test_non_native_byte_order_economic_rank_1(self):
        self.base_non_simple_strides(make_nonnative, 'economic', 1, False)

    def test_non_native_byte_order_rank_p(self):
        self.base_non_simple_strides(make_nonnative, 'full', 3, False)

    def test_non_native_byte_order_economic_rank_p(self):
        self.base_non_simple_strides(make_nonnative, 'economic', 3, False)

    def test_overwrite_qruv_rank_1(self):
        a, q0, r0, u0, v0 = self.generate('sqr')
        a1 = a + np.outer(u0, v0.conj())
        q = q0.copy('F')
        r = r0.copy('F')
        u = u0.copy('F')
        v = v0.copy('F')
        q1, r1 = qr_update(q, r, u, v, False)
        check_qr(q1, r1, a1, self.rtol, self.atol)
        check_qr(q, r, a, self.rtol, self.atol)
        q2, r2 = qr_update(q, r, u, v, True)
        check_qr(q2, r2, a1, self.rtol, self.atol)
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r2, r, rtol=self.rtol, atol=self.atol)
        q = q0.copy('C')
        r = r0.copy('C')
        u = u0.copy('C')
        v = v0.copy('C')
        q3, r3 = qr_update(q, r, u, v, True)
        check_qr(q3, r3, a1, self.rtol, self.atol)
        assert_allclose(q3, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r3, r, rtol=self.rtol, atol=self.atol)

    def test_overwrite_qruv_rank_1_economic(self):
        a, q0, r0, u0, v0 = self.generate('tall', 'economic')
        a1 = a + np.outer(u0, v0.conj())
        q = q0.copy('F')
        r = r0.copy('F')
        u = u0.copy('F')
        v = v0.copy('F')
        q1, r1 = qr_update(q, r, u, v, False)
        check_qr(q1, r1, a1, self.rtol, self.atol, False)
        check_qr(q, r, a, self.rtol, self.atol, False)
        q2, r2 = qr_update(q, r, u, v, True)
        check_qr(q2, r2, a1, self.rtol, self.atol, False)
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r2, r, rtol=self.rtol, atol=self.atol)
        q = q0.copy('C')
        r = r0.copy('C')
        u = u0.copy('C')
        v = v0.copy('C')
        q3, r3 = qr_update(q, r, u, v, True)
        check_qr(q3, r3, a1, self.rtol, self.atol, False)
        assert_allclose(q3, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r3, r, rtol=self.rtol, atol=self.atol)

    def test_overwrite_qruv_rank_p(self):
        a, q0, r0, u0, v0 = self.generate('sqr', p=3)
        a1 = a + np.dot(u0, v0.T.conj())
        q = q0.copy('F')
        r = r0.copy('F')
        u = u0.copy('F')
        v = v0.copy('C')
        q1, r1 = qr_update(q, r, u, v, False)
        check_qr(q1, r1, a1, self.rtol, self.atol)
        check_qr(q, r, a, self.rtol, self.atol)
        q2, r2 = qr_update(q, r, u, v, True)
        check_qr(q2, r2, a1, self.rtol, self.atol)
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r2, r, rtol=self.rtol, atol=self.atol)

    def test_empty_inputs(self):
        a, q, r, u, v = self.generate('tall')
        assert_raises(ValueError, qr_update, np.array([]), r, u, v)
        assert_raises(ValueError, qr_update, q, np.array([]), u, v)
        assert_raises(ValueError, qr_update, q, r, np.array([]), v)
        assert_raises(ValueError, qr_update, q, r, u, np.array([]))

    def test_mismatched_shapes(self):
        a, q, r, u, v = self.generate('tall')
        assert_raises(ValueError, qr_update, q, r[1:], u, v)
        assert_raises(ValueError, qr_update, q[:-2], r, u, v)
        assert_raises(ValueError, qr_update, q, r, u[1:], v)
        assert_raises(ValueError, qr_update, q, r, u, v[1:])

    def test_unsupported_dtypes(self):
        dts = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'longdouble', 'clongdouble', 'bool']
        a, q0, r0, u0, v0 = self.generate('tall')
        for dtype in dts:
            q = q0.real.astype(dtype)
            with np.errstate(invalid='ignore'):
                r = r0.real.astype(dtype)
            u = u0.real.astype(dtype)
            v = v0.real.astype(dtype)
            assert_raises(ValueError, qr_update, q, r0, u0, v0)
            assert_raises(ValueError, qr_update, q0, r, u0, v0)
            assert_raises(ValueError, qr_update, q0, r0, u, v0)
            assert_raises(ValueError, qr_update, q0, r0, u0, v)

    def test_integer_input(self):
        q = np.arange(16).reshape(4, 4)
        r = q.copy()
        u = q[:, 0].copy()
        v = r[0, :].copy()
        assert_raises(ValueError, qr_update, q, r, u, v)

    def test_check_finite(self):
        a0, q0, r0, u0, v0 = self.generate('tall', p=3)
        q = q0.copy('F')
        q[1, 1] = np.nan
        assert_raises(ValueError, qr_update, q, r0, u0[:, 0], v0[:, 0])
        assert_raises(ValueError, qr_update, q, r0, u0, v0)
        r = r0.copy('F')
        r[1, 1] = np.nan
        assert_raises(ValueError, qr_update, q0, r, u0[:, 0], v0[:, 0])
        assert_raises(ValueError, qr_update, q0, r, u0, v0)
        u = u0.copy('F')
        u[0, 0] = np.nan
        assert_raises(ValueError, qr_update, q0, r0, u[:, 0], v0[:, 0])
        assert_raises(ValueError, qr_update, q0, r0, u, v0)
        v = v0.copy('F')
        v[0, 0] = np.nan
        assert_raises(ValueError, qr_update, q0, r0, u[:, 0], v[:, 0])
        assert_raises(ValueError, qr_update, q0, r0, u, v)

    def test_economic_check_finite(self):
        a0, q0, r0, u0, v0 = self.generate('tall', mode='economic', p=3)
        q = q0.copy('F')
        q[1, 1] = np.nan
        assert_raises(ValueError, qr_update, q, r0, u0[:, 0], v0[:, 0])
        assert_raises(ValueError, qr_update, q, r0, u0, v0)
        r = r0.copy('F')
        r[1, 1] = np.nan
        assert_raises(ValueError, qr_update, q0, r, u0[:, 0], v0[:, 0])
        assert_raises(ValueError, qr_update, q0, r, u0, v0)
        u = u0.copy('F')
        u[0, 0] = np.nan
        assert_raises(ValueError, qr_update, q0, r0, u[:, 0], v0[:, 0])
        assert_raises(ValueError, qr_update, q0, r0, u, v0)
        v = v0.copy('F')
        v[0, 0] = np.nan
        assert_raises(ValueError, qr_update, q0, r0, u[:, 0], v[:, 0])
        assert_raises(ValueError, qr_update, q0, r0, u, v)

    def test_u_exactly_in_span_q(self):
        q = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], self.dtype)
        r = np.array([[1, 0], [0, 1]], self.dtype)
        u = np.array([0, 0, 0, -1], self.dtype)
        v = np.array([1, 2], self.dtype)
        q1, r1 = qr_update(q, r, u, v)
        a1 = np.dot(q, r) + np.outer(u, v.conj())
        check_qr(q1, r1, a1, self.rtol, self.atol, False)