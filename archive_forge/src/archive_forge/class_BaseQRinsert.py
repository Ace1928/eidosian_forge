import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
class BaseQRinsert(BaseQRdeltas):

    def generate(self, type, mode='full', which='row', p=1):
        a, q, r = super().generate(type, mode)
        assert_(p > 0)
        if which == 'row':
            if p == 1:
                u = np.random.random(a.shape[1])
            else:
                u = np.random.random((p, a.shape[1]))
        elif which == 'col':
            if p == 1:
                u = np.random.random(a.shape[0])
            else:
                u = np.random.random((a.shape[0], p))
        else:
            ValueError('which should be either "row" or "col"')
        if np.iscomplexobj(self.dtype.type(1)):
            b = np.random.random(u.shape)
            u = u + 1j * b
        u = u.astype(self.dtype)
        return (a, q, r, u)

    def test_sqr_1_row(self):
        a, q, r, u = self.generate('sqr', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_sqr_p_row(self):
        a, q, r, u = self.generate('sqr', which='row', p=3)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_sqr_1_col(self):
        a, q, r, u = self.generate('sqr', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_sqr_p_col(self):
        a, q, r, u = self.generate('sqr', which='col', p=3)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_tall_1_row(self):
        a, q, r, u = self.generate('tall', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_tall_p_row(self):
        a, q, r, u = self.generate('tall', which='row', p=3)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_tall_1_col(self):
        a, q, r, u = self.generate('tall', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def base_tall_p_col_xxx(self, p):
        a, q, r, u = self.generate('tall', which='col', p=p)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(p, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_tall_p_col_tall(self):
        self.base_tall_p_col_xxx(3)

    def test_tall_p_col_sqr(self):
        self.base_tall_p_col_xxx(5)

    def test_tall_p_col_fat(self):
        self.base_tall_p_col_xxx(7)

    def test_fat_1_row(self):
        a, q, r, u = self.generate('fat', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def base_fat_p_row_xxx(self, p):
        a, q, r, u = self.generate('fat', which='row', p=p)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, np.full(p, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_fat_p_row_fat(self):
        self.base_fat_p_row_xxx(3)

    def test_fat_p_row_sqr(self):
        self.base_fat_p_row_xxx(5)

    def test_fat_p_row_tall(self):
        self.base_fat_p_row_xxx(7)

    def test_fat_1_col(self):
        a, q, r, u = self.generate('fat', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_fat_p_col(self):
        a, q, r, u = self.generate('fat', which='col', p=3)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_economic_1_row(self):
        a, q, r, u = self.generate('tall', 'economic', 'row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row, overwrite_qru=False)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_economic_p_row(self):
        a, q, r, u = self.generate('tall', 'economic', 'row', 3)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row, overwrite_qru=False)
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_economic_1_col(self):
        a, q, r, u = self.generate('tall', 'economic', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u.copy(), col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_economic_1_col_bad_update(self):
        q = np.eye(5, 3, dtype=self.dtype)
        r = np.eye(3, dtype=self.dtype)
        u = np.array([1, 0, 0, 0, 0], self.dtype)
        assert_raises(linalg.LinAlgError, qr_insert, q, r, u, 0, 'col')

    def base_economic_p_col_xxx(self, p):
        a, q, r, u = self.generate('tall', 'economic', which='col', p=p)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(p, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_economic_p_col_eco(self):
        self.base_economic_p_col_xxx(3)

    def test_economic_p_col_sqr(self):
        self.base_economic_p_col_xxx(5)

    def test_economic_p_col_fat(self):
        self.base_economic_p_col_xxx(7)

    def test_Mx1_1_row(self):
        a, q, r, u = self.generate('Mx1', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_Mx1_p_row(self):
        a, q, r, u = self.generate('Mx1', which='row', p=3)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_Mx1_1_col(self):
        a, q, r, u = self.generate('Mx1', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_Mx1_p_col(self):
        a, q, r, u = self.generate('Mx1', which='col', p=3)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_Mx1_economic_1_row(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_Mx1_economic_p_row(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'row', 3)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_Mx1_economic_1_col(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_Mx1_economic_p_col(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'col', 3)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    def test_1xN_1_row(self):
        a, q, r, u = self.generate('1xN', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1xN_p_row(self):
        a, q, r, u = self.generate('1xN', which='row', p=3)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1xN_1_col(self):
        a, q, r, u = self.generate('1xN', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1xN_p_col(self):
        a, q, r, u = self.generate('1xN', which='col', p=3)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_1_row(self):
        a, q, r, u = self.generate('1x1', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, row, u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_p_row(self):
        a, q, r, u = self.generate('1x1', which='row', p=3)
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_1_col(self):
        a, q, r, u = self.generate('1x1', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, col, u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_p_col(self):
        a, q, r, u = self.generate('1x1', which='col', p=3)
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    def test_1x1_1_scalar(self):
        a, q, r, u = self.generate('1x1', which='row')
        assert_raises(ValueError, qr_insert, q[0, 0], r, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r[0, 0], u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r, u[0], 0, 'row')
        assert_raises(ValueError, qr_insert, q[0, 0], r, u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r[0, 0], u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r, u[0], 0, 'col')

    def base_non_simple_strides(self, adjust_strides, k, p, which):
        for type in ['sqr', 'tall', 'fat']:
            a, q0, r0, u0 = self.generate(type, which=which, p=p)
            qs, rs, us = adjust_strides((q0, r0, u0))
            if p == 1:
                ai = np.insert(a, k, u0, 0 if which == 'row' else 1)
            else:
                ai = np.insert(a, np.full(p, k, np.intp), u0 if which == 'row' else u0, 0 if which == 'row' else 1)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            q1, r1 = qr_insert(qs, r, u, k, which, overwrite_qru=False)
            check_qr(q1, r1, ai, self.rtol, self.atol)
            q1o, r1o = qr_insert(qs, r, u, k, which, overwrite_qru=True)
            check_qr(q1o, r1o, ai, self.rtol, self.atol)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            q2, r2 = qr_insert(q, rs, u, k, which, overwrite_qru=False)
            check_qr(q2, r2, ai, self.rtol, self.atol)
            q2o, r2o = qr_insert(q, rs, u, k, which, overwrite_qru=True)
            check_qr(q2o, r2o, ai, self.rtol, self.atol)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            q3, r3 = qr_insert(q, r, us, k, which, overwrite_qru=False)
            check_qr(q3, r3, ai, self.rtol, self.atol)
            q3o, r3o = qr_insert(q, r, us, k, which, overwrite_qru=True)
            check_qr(q3o, r3o, ai, self.rtol, self.atol)
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            qs, rs, us = adjust_strides((q, r, u))
            q5, r5 = qr_insert(qs, rs, us, k, which, overwrite_qru=False)
            check_qr(q5, r5, ai, self.rtol, self.atol)
            q5o, r5o = qr_insert(qs, rs, us, k, which, overwrite_qru=True)
            check_qr(q5o, r5o, ai, self.rtol, self.atol)

    def test_non_unit_strides_1_row(self):
        self.base_non_simple_strides(make_strided, 0, 1, 'row')

    def test_non_unit_strides_p_row(self):
        self.base_non_simple_strides(make_strided, 0, 3, 'row')

    def test_non_unit_strides_1_col(self):
        self.base_non_simple_strides(make_strided, 0, 1, 'col')

    def test_non_unit_strides_p_col(self):
        self.base_non_simple_strides(make_strided, 0, 3, 'col')

    def test_neg_strides_1_row(self):
        self.base_non_simple_strides(negate_strides, 0, 1, 'row')

    def test_neg_strides_p_row(self):
        self.base_non_simple_strides(negate_strides, 0, 3, 'row')

    def test_neg_strides_1_col(self):
        self.base_non_simple_strides(negate_strides, 0, 1, 'col')

    def test_neg_strides_p_col(self):
        self.base_non_simple_strides(negate_strides, 0, 3, 'col')

    def test_non_itemsize_strides_1_row(self):
        self.base_non_simple_strides(nonitemsize_strides, 0, 1, 'row')

    def test_non_itemsize_strides_p_row(self):
        self.base_non_simple_strides(nonitemsize_strides, 0, 3, 'row')

    def test_non_itemsize_strides_1_col(self):
        self.base_non_simple_strides(nonitemsize_strides, 0, 1, 'col')

    def test_non_itemsize_strides_p_col(self):
        self.base_non_simple_strides(nonitemsize_strides, 0, 3, 'col')

    def test_non_native_byte_order_1_row(self):
        self.base_non_simple_strides(make_nonnative, 0, 1, 'row')

    def test_non_native_byte_order_p_row(self):
        self.base_non_simple_strides(make_nonnative, 0, 3, 'row')

    def test_non_native_byte_order_1_col(self):
        self.base_non_simple_strides(make_nonnative, 0, 1, 'col')

    def test_non_native_byte_order_p_col(self):
        self.base_non_simple_strides(make_nonnative, 0, 3, 'col')

    def test_overwrite_qu_rank_1(self):
        a, q0, r, u = self.generate('sqr', which='col', p=1)
        q = q0.copy('C')
        u0 = u.copy()
        q1, r1 = qr_insert(q, r, u, 0, 'col', overwrite_qru=False)
        a1 = np.insert(a, 0, u0, 1)
        check_qr(q1, r1, a1, self.rtol, self.atol)
        check_qr(q, r, a, self.rtol, self.atol)
        q2, r2 = qr_insert(q, r, u, 0, 'col', overwrite_qru=True)
        check_qr(q2, r2, a1, self.rtol, self.atol)
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(u, u0.conj(), self.rtol, self.atol)
        qF = q0.copy('F')
        u1 = u0.copy()
        q3, r3 = qr_insert(qF, r, u1, 0, 'col', overwrite_qru=False)
        check_qr(q3, r3, a1, self.rtol, self.atol)
        check_qr(qF, r, a, self.rtol, self.atol)
        q4, r4 = qr_insert(qF, r, u1, 0, 'col', overwrite_qru=True)
        check_qr(q4, r4, a1, self.rtol, self.atol)
        assert_allclose(q4, qF, rtol=self.rtol, atol=self.atol)

    def test_overwrite_qu_rank_p(self):
        a, q0, r, u = self.generate('sqr', which='col', p=3)
        q = q0.copy('F')
        a1 = np.insert(a, np.zeros(3, np.intp), u, 1)
        q1, r1 = qr_insert(q, r, u, 0, 'col', overwrite_qru=False)
        check_qr(q1, r1, a1, self.rtol, self.atol)
        check_qr(q, r, a, self.rtol, self.atol)
        q2, r2 = qr_insert(q, r, u, 0, 'col', overwrite_qru=True)
        check_qr(q2, r2, a1, self.rtol, self.atol)
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)

    def test_empty_inputs(self):
        a, q, r, u = self.generate('sqr', which='row')
        assert_raises(ValueError, qr_insert, np.array([]), r, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, np.array([]), u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r, np.array([]), 0, 'row')
        assert_raises(ValueError, qr_insert, np.array([]), r, u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, np.array([]), u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r, np.array([]), 0, 'col')

    def test_mismatched_shapes(self):
        a, q, r, u = self.generate('tall', which='row')
        assert_raises(ValueError, qr_insert, q, r[1:], u, 0, 'row')
        assert_raises(ValueError, qr_insert, q[:-2], r, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r, u[1:], 0, 'row')
        assert_raises(ValueError, qr_insert, q, r[1:], u, 0, 'col')
        assert_raises(ValueError, qr_insert, q[:-2], r, u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r, u[1:], 0, 'col')

    def test_unsupported_dtypes(self):
        dts = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'longdouble', 'clongdouble', 'bool']
        a, q0, r0, u0 = self.generate('sqr', which='row')
        for dtype in dts:
            q = q0.real.astype(dtype)
            with np.errstate(invalid='ignore'):
                r = r0.real.astype(dtype)
            u = u0.real.astype(dtype)
            assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'row')
            assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'col')
            assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'row')
            assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'col')
            assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'row')
            assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'col')

    def test_check_finite(self):
        a0, q0, r0, u0 = self.generate('sqr', which='row', p=3)
        q = q0.copy('F')
        q[1, 1] = np.nan
        assert_raises(ValueError, qr_insert, q, r0, u0[:, 0], 0, 'row')
        assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r0, u0[:, 0], 0, 'col')
        assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'col')
        r = r0.copy('F')
        r[1, 1] = np.nan
        assert_raises(ValueError, qr_insert, q0, r, u0[:, 0], 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r, u0[:, 0], 0, 'col')
        assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'col')
        u = u0.copy('F')
        u[0, 0] = np.nan
        assert_raises(ValueError, qr_insert, q0, r0, u[:, 0], 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r0, u[:, 0], 0, 'col')
        assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'col')