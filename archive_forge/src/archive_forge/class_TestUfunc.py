import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
class TestUfunc:

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_(pickle.loads(pickle.dumps(np.sin, protocol=proto)) is np.sin)
            res = pickle.loads(pickle.dumps(_rational_tests.test_add, protocol=proto))
            assert_(res is _rational_tests.test_add)

    def test_pickle_withstring(self):
        astring = b"cnumpy.core\n_ufunc_reconstruct\np0\n(S'numpy.core.umath'\np1\nS'cos'\np2\ntp3\nRp4\n."
        assert_(pickle.loads(astring) is np.cos)

    @pytest.mark.skipif(IS_PYPY, reason="'is' check does not work on PyPy")
    def test_pickle_name_is_qualname(self):
        _pickleable_module_global.ufunc = umt._pickleable_module_global_ufunc
        obj = pickle.loads(pickle.dumps(_pickleable_module_global.ufunc))
        assert obj is umt._pickleable_module_global_ufunc

    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()
        assert_array_equal(np.add.reduceat(x, idx)[::2], [1, 3, 5, 7])

    def test_all_ufunc(self):
        """Try to check presence and results of all ufuncs.

        The list of ufuncs comes from generate_umath.py and is as follows:

        =====  ====  =============  ===============  ========================
        done   args   function        types                notes
        =====  ====  =============  ===============  ========================
        n      1     conjugate      nums + O
        n      1     absolute       nums + O         complex -> real
        n      1     negative       nums + O
        n      1     sign           nums + O         -> int
        n      1     invert         bool + ints + O  flts raise an error
        n      1     degrees        real + M         cmplx raise an error
        n      1     radians        real + M         cmplx raise an error
        n      1     arccos         flts + M
        n      1     arccosh        flts + M
        n      1     arcsin         flts + M
        n      1     arcsinh        flts + M
        n      1     arctan         flts + M
        n      1     arctanh        flts + M
        n      1     cos            flts + M
        n      1     sin            flts + M
        n      1     tan            flts + M
        n      1     cosh           flts + M
        n      1     sinh           flts + M
        n      1     tanh           flts + M
        n      1     exp            flts + M
        n      1     expm1          flts + M
        n      1     log            flts + M
        n      1     log10          flts + M
        n      1     log1p          flts + M
        n      1     sqrt           flts + M         real x < 0 raises error
        n      1     ceil           real + M
        n      1     trunc          real + M
        n      1     floor          real + M
        n      1     fabs           real + M
        n      1     rint           flts + M
        n      1     isnan          flts             -> bool
        n      1     isinf          flts             -> bool
        n      1     isfinite       flts             -> bool
        n      1     signbit        real             -> bool
        n      1     modf           real             -> (frac, int)
        n      1     logical_not    bool + nums + M  -> bool
        n      2     left_shift     ints + O         flts raise an error
        n      2     right_shift    ints + O         flts raise an error
        n      2     add            bool + nums + O  boolean + is ||
        n      2     subtract       bool + nums + O  boolean - is ^
        n      2     multiply       bool + nums + O  boolean * is &
        n      2     divide         nums + O
        n      2     floor_divide   nums + O
        n      2     true_divide    nums + O         bBhH -> f, iIlLqQ -> d
        n      2     fmod           nums + M
        n      2     power          nums + O
        n      2     greater        bool + nums + O  -> bool
        n      2     greater_equal  bool + nums + O  -> bool
        n      2     less           bool + nums + O  -> bool
        n      2     less_equal     bool + nums + O  -> bool
        n      2     equal          bool + nums + O  -> bool
        n      2     not_equal      bool + nums + O  -> bool
        n      2     logical_and    bool + nums + M  -> bool
        n      2     logical_or     bool + nums + M  -> bool
        n      2     logical_xor    bool + nums + M  -> bool
        n      2     maximum        bool + nums + O
        n      2     minimum        bool + nums + O
        n      2     bitwise_and    bool + ints + O  flts raise an error
        n      2     bitwise_or     bool + ints + O  flts raise an error
        n      2     bitwise_xor    bool + ints + O  flts raise an error
        n      2     arctan2        real + M
        n      2     remainder      ints + real + O
        n      2     hypot          real + M
        =====  ====  =============  ===============  ========================

        Types other than those listed will be accepted, but they are cast to
        the smallest compatible type for which the function is defined. The
        casting rules are:

        bool -> int8 -> float32
        ints -> double

        """
        pass
    size_inferred = 2
    can_ignore = 4

    def test_signature0(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(2, 1, '(i),(i)->()')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 1, 0))
        assert_equal(ixs, (0, 0))
        assert_equal(flags, (self.size_inferred,))
        assert_equal(sizes, (-1,))

    def test_signature1(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(2, 1, '(),()->()')
        assert_equal(enabled, 0)
        assert_equal(num_dims, (0, 0, 0))
        assert_equal(ixs, ())
        assert_equal(flags, ())
        assert_equal(sizes, ())

    def test_signature2(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(2, 1, '(i1,i2),(J_1)->(_kAB)')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 1))
        assert_equal(ixs, (0, 1, 2, 3))
        assert_equal(flags, (self.size_inferred,) * 4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature3(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(2, 1, '(i1, i12),   (J_1)->(i12, i2)')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 2))
        assert_equal(ixs, (0, 1, 2, 1, 3))
        assert_equal(flags, (self.size_inferred,) * 4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature4(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(2, 1, '(n,k),(k,m)->(n,m)')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred,) * 3)
        assert_equal(sizes, (-1, -1, -1))

    def test_signature5(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(2, 1, '(n?,k),(k,m?)->(n?,m?)')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred | self.can_ignore, self.size_inferred, self.size_inferred | self.can_ignore))
        assert_equal(sizes, (-1, -1, -1))

    def test_signature6(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(1, 1, '(3)->()')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature7(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(3, 1, '(3),(03,3),(n)->(9)')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (0, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature8(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(3, 1, '(3?),(3?,3?),(n)->(9)')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature9(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(1, 1, '(  3)  -> ( )')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature10(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(3, 1, '( 3? ) , (3? ,  3?) ,(n )-> ( 9)')
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature_failure_extra_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, '((i)),(i)->()')

    def test_signature_failure_mismatching_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, '(i),)i(->()')

    def test_signature_failure_signature_missing_input_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, '(i),->()')

    def test_signature_failure_signature_missing_output_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 2, '(i),(i)->()')

    def test_get_signature(self):
        assert_equal(umt.inner1d.signature, '(i),(i)->()')

    def test_forced_sig(self):
        a = 0.5 * np.arange(3, dtype='f8')
        assert_equal(np.add(a, 0.5), [0.5, 1, 1.5])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig='i', casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig='ii->i', casting='unsafe'), [0, 0, 1])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig=('i4',), casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig=('i4', 'i4', 'i4'), casting='unsafe'), [0, 0, 1])
        b = np.zeros((3,), dtype='f8')
        np.add(a, 0.5, out=b)
        assert_equal(b, [0.5, 1, 1.5])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig='i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig='ii->i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig=('i4',), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig=('i4', 'i4', 'i4'), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])

    def test_signature_all_None(self):
        res1 = np.add([3], [4], sig=(None, None, None))
        res2 = np.add([3], [4])
        assert_array_equal(res1, res2)
        res1 = np.maximum([3], [4], sig=(None, None, None))
        res2 = np.maximum([3], [4])
        assert_array_equal(res1, res2)
        with pytest.raises(TypeError):
            np.add(3, 4, signature=(None,))

    def test_signature_dtype_type(self):
        float_dtype = type(np.dtype(np.float64))
        np.add(3, 4, signature=(float_dtype, float_dtype, None))

    @pytest.mark.parametrize('get_kwarg', [lambda dt: dict(dtype=x), lambda dt: dict(signature=(x, None, None))])
    def test_signature_dtype_instances_allowed(self, get_kwarg):
        int64 = np.dtype('int64')
        int64_2 = pickle.loads(pickle.dumps(int64))
        assert int64 is not int64_2
        assert np.add(1, 2, **get_kwarg(int64_2)).dtype == int64
        td = np.timedelta(2, 's')
        assert np.add(td, td, **get_kwarg('m8')).dtype == 'm8[s]'

    @pytest.mark.parametrize('get_kwarg', [param(lambda x: dict(dtype=x), id='dtype'), param(lambda x: dict(signature=(x, None, None)), id='signature')])
    def test_signature_dtype_instances_allowed(self, get_kwarg):
        msg = 'The `dtype` and `signature` arguments to ufuncs'
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype('int64').newbyteorder()))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg(np.dtype('m8[ns]')))
        with pytest.raises(TypeError, match=msg):
            np.add(3, 5, **get_kwarg('m8[ns]'))

    @pytest.mark.parametrize('casting', ['unsafe', 'same_kind', 'safe'])
    def test_partial_signature_mismatch(self, casting):
        res = np.ldexp(np.float32(1.0), np.int_(2), dtype='d')
        assert res.dtype == 'd'
        res = np.ldexp(np.float32(1.0), np.int_(2), signature=(None, None, 'd'))
        assert res.dtype == 'd'
        with pytest.raises(TypeError):
            np.ldexp(1.0, np.uint64(3), dtype='d')
        with pytest.raises(TypeError):
            np.ldexp(1.0, np.uint64(3), signature=(None, None, 'd'))

    def test_partial_signature_mismatch_with_cache(self):
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=('e', 'd', None))
        np.add(np.float16(1), np.float64(2))
        with pytest.raises(TypeError):
            np.add(np.float16(1), np.uint64(2), sig=('e', 'd', None))

    def test_use_output_signature_for_all_arguments(self):
        res = np.power(1.5, 2.8, dtype=np.intp, casting='unsafe')
        assert res == 1
        res = np.power(1.5, 2.8, signature=(None, None, np.intp), casting='unsafe')
        assert res == 1
        with pytest.raises(TypeError):
            np.power(1.5, 2.8, dtype=np.intp)

    def test_signature_errors(self):
        with pytest.raises(TypeError, match='the signature object to ufunc must be a string or'):
            np.add(3, 4, signature=123.0)
        with pytest.raises(ValueError):
            np.add(3, 4, signature='%^->#')
        with pytest.raises(ValueError):
            np.add(3, 4, signature=b'ii-i')
        with pytest.raises(ValueError):
            np.add(3, 4, signature='ii>i')
        with pytest.raises(ValueError):
            np.add(3, 4, signature=(None, 'f8'))
        with pytest.raises(UnicodeDecodeError):
            np.add(3, 4, signature=b'\xff\xff->i')

    def test_forced_dtype_times(self):
        a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='>M8[D]')
        np.maximum(a, a, dtype='M')
        np.maximum.reduce(a, dtype='M')
        arr = np.arange(10, dtype='m8[s]')
        np.add(arr, arr, dtype='m')
        np.maximum(arr, arr, dtype='m')

    @pytest.mark.parametrize('ufunc', [np.add, np.sqrt])
    def test_cast_safety(self, ufunc):
        """Basic test for the safest casts, because ufuncs inner loops can
        indicate a cast-safety as well (which is normally always "no").
        """

        def call_ufunc(arr, **kwargs):
            return ufunc(*(arr,) * ufunc.nin, **kwargs)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr_bs = arr.astype(arr.dtype.newbyteorder())
        expected = call_ufunc(arr)
        res = call_ufunc(arr, casting='no')
        assert_array_equal(expected, res)
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, casting='no')
        res = call_ufunc(arr_bs, casting='equiv')
        assert_array_equal(expected, res)
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, dtype=np.float64, casting='equiv')
        res = call_ufunc(arr_bs, dtype=np.float64, casting='safe')
        expected = call_ufunc(arr.astype(np.float64))
        assert_array_equal(expected, res)

    def test_true_divide(self):
        a = np.array(10)
        b = np.array(20)
        tgt = np.array(0.5)
        for tc in 'bhilqBHILQefdgFDG':
            dt = np.dtype(tc)
            aa = a.astype(dt)
            bb = b.astype(dt)
            for x, y in itertools.product([aa, -aa], [bb, -bb]):
                if tc in 'FDG':
                    tgt = complex(x) / complex(y)
                else:
                    tgt = float(x) / float(y)
                res = np.true_divide(x, y)
                rtol = max(np.finfo(res).resolution, 1e-15)
                assert_allclose(res, tgt, rtol=rtol)
                if tc in 'bhilqBHILQ':
                    assert_(res.dtype.name == 'float64')
                else:
                    assert_(res.dtype.name == dt.name)
                for tcout in 'bhilqBHILQ':
                    dtout = np.dtype(tcout)
                    assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)
                for tcout in 'efdg':
                    dtout = np.dtype(tcout)
                    if tc in 'FDG':
                        assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)
                    else:
                        tgt = float(x) / float(y)
                        rtol = max(np.finfo(dtout).resolution, 1e-15)
                        with suppress_warnings() as sup:
                            sup.filter(UserWarning)
                            if not np.isnan(np.finfo(dtout).tiny):
                                atol = max(np.finfo(dtout).tiny, 3e-308)
                            else:
                                atol = 3e-308
                        with np.errstate(invalid='ignore', over='ignore'):
                            res = np.true_divide(x, y, dtype=dtout)
                        if not np.isfinite(res) and tcout == 'e':
                            continue
                        assert_allclose(res, tgt, rtol=rtol, atol=atol)
                        assert_(res.dtype.name == dtout.name)
                for tcout in 'FDG':
                    dtout = np.dtype(tcout)
                    tgt = complex(x) / complex(y)
                    rtol = max(np.finfo(dtout).resolution, 1e-15)
                    with suppress_warnings() as sup:
                        sup.filter(UserWarning)
                        if not np.isnan(np.finfo(dtout).tiny):
                            atol = max(np.finfo(dtout).tiny, 3e-308)
                        else:
                            atol = 3e-308
                    res = np.true_divide(x, y, dtype=dtout)
                    if not np.isfinite(res):
                        continue
                    assert_allclose(res, tgt, rtol=rtol, atol=atol)
                    assert_(res.dtype.name == dtout.name)
        a = np.ones((), dtype=np.bool_)
        res = np.true_divide(a, a)
        assert_(res == 1.0)
        assert_(res.dtype.name == 'float64')
        res = np.true_divide(~a, a)
        assert_(res == 0.0)
        assert_(res.dtype.name == 'float64')

    def test_sum_stability(self):
        a = np.ones(500, dtype=np.float32)
        assert_almost_equal((a / 10.0).sum() - a.size / 10.0, 0, 4)
        a = np.ones(500, dtype=np.float64)
        assert_almost_equal((a / 10.0).sum() - a.size / 10.0, 0, 13)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sum(self):
        for dt in (int, np.float16, np.float32, np.float64, np.longdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', RuntimeWarning)
                    tgt = dt(v * (v + 1) / 2)
                    overflow = not np.isfinite(tgt)
                    assert_equal(len(w), 1 * overflow)
                    d = np.arange(1, v + 1, dtype=dt)
                    assert_almost_equal(np.sum(d), tgt)
                    assert_equal(len(w), 2 * overflow)
                    assert_almost_equal(np.sum(d[::-1]), tgt)
                    assert_equal(len(w), 3 * overflow)
            d = np.ones(500, dtype=dt)
            assert_almost_equal(np.sum(d[::2]), 250.0)
            assert_almost_equal(np.sum(d[1::2]), 250.0)
            assert_almost_equal(np.sum(d[::3]), 167.0)
            assert_almost_equal(np.sum(d[1::3]), 167.0)
            assert_almost_equal(np.sum(d[::-2]), 250.0)
            assert_almost_equal(np.sum(d[-1::-2]), 250.0)
            assert_almost_equal(np.sum(d[::-3]), 167.0)
            assert_almost_equal(np.sum(d[-1::-3]), 167.0)
            d = np.ones((1,), dtype=dt)
            d += d
            assert_almost_equal(d, 2.0)

    def test_sum_complex(self):
        for dt in (np.complex64, np.complex128, np.clongdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
                tgt = dt(v * (v + 1) / 2) - dt(v * (v + 1) / 2 * 1j)
                d = np.empty(v, dtype=dt)
                d.real = np.arange(1, v + 1)
                d.imag = -np.arange(1, v + 1)
                assert_almost_equal(np.sum(d), tgt)
                assert_almost_equal(np.sum(d[::-1]), tgt)
            d = np.ones(500, dtype=dt) + 1j
            assert_almost_equal(np.sum(d[::2]), 250.0 + 250j)
            assert_almost_equal(np.sum(d[1::2]), 250.0 + 250j)
            assert_almost_equal(np.sum(d[::3]), 167.0 + 167j)
            assert_almost_equal(np.sum(d[1::3]), 167.0 + 167j)
            assert_almost_equal(np.sum(d[::-2]), 250.0 + 250j)
            assert_almost_equal(np.sum(d[-1::-2]), 250.0 + 250j)
            assert_almost_equal(np.sum(d[::-3]), 167.0 + 167j)
            assert_almost_equal(np.sum(d[-1::-3]), 167.0 + 167j)
            d = np.ones((1,), dtype=dt) + 1j
            d += d
            assert_almost_equal(d, 2.0 + 2j)

    def test_sum_initial(self):
        assert_equal(np.sum([3], initial=2), 5)
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)
        assert_equal(np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2), [12, 12, 12])

    def test_sum_where(self):
        assert_equal(np.sum([[1.0, 2.0], [3.0, 4.0]], where=[True, False]), 4.0)
        assert_equal(np.sum([[1.0, 2.0], [3.0, 4.0]], axis=0, initial=5.0, where=[True, False]), [9.0, 5.0])

    def test_inner1d(self):
        a = np.arange(6).reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a * a, axis=-1))
        a = np.arange(6)
        assert_array_equal(umt.inner1d(a, a), np.sum(a * a))

    def test_broadcast(self):
        msg = 'broadcast'
        a = np.arange(4).reshape((2, 1, 2))
        b = np.arange(4).reshape((1, 2, 2))
        assert_array_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        msg = 'extend & broadcast loop dimensions'
        b = np.arange(4).reshape((2, 2))
        assert_array_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        a = np.arange(8).reshape((4, 2))
        b = np.arange(4).reshape((4, 1))
        assert_raises(ValueError, umt.inner1d, a, b)
        a = np.arange(8).reshape((4, 2))
        b = np.array(7)
        assert_raises(ValueError, umt.inner1d, a, b)
        a = np.arange(2).reshape((2, 1, 1))
        b = np.arange(3).reshape((3, 1, 1))
        assert_raises(ValueError, umt.inner1d, a, b)
        a = np.arange(2)
        b = np.arange(4).reshape((2, 2))
        u, v = np.broadcast_arrays(a, b)
        assert_equal(u.strides[0], 0)
        x = u + v
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            u += v
            assert_equal(len(w), 1)
            assert_(x[0, 0] != u[0, 0])
        a = np.arange(6).reshape(3, 2)
        b = np.ones(2)
        out = np.empty(())
        assert_raises(ValueError, umt.inner1d, a, b, out)
        out2 = np.empty(3)
        c = umt.inner1d(a, b, out2)
        assert_(c is out2)

    def test_out_broadcasts(self):
        arr = np.arange(3).reshape(1, 3)
        out = np.empty((5, 4, 3))
        np.add(arr, arr, out=out)
        assert (out == np.arange(3) * 2).all()
        umt.inner1d(arr, arr, out=out)
        assert (out == 5).all()

    @pytest.mark.parametrize(['arr', 'out'], [([2], np.empty(())), ([1, 2], np.empty(1)), (np.ones((4, 3)), np.empty((4, 1)))], ids=['(1,)->()', '(2,)->(1,)', '(4, 3)->(4, 1)'])
    def test_out_broadcast_errors(self, arr, out):
        with pytest.raises(ValueError, match='non-broadcastable'):
            np.positive(arr, out=out)
        with pytest.raises(ValueError, match='non-broadcastable'):
            np.add(np.ones(()), arr, out=out)

    def test_type_cast(self):
        msg = 'type cast'
        a = np.arange(6, dtype='short').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a * a, axis=-1), err_msg=msg)
        msg = 'type cast on one argument'
        a = np.arange(6).reshape((2, 3))
        b = a + 0.1
        assert_array_almost_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)

    def test_endian(self):
        msg = 'big endian'
        a = np.arange(6, dtype='>i4').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a * a, axis=-1), err_msg=msg)
        msg = 'little endian'
        a = np.arange(6, dtype='<i4').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a * a, axis=-1), err_msg=msg)
        Ba = np.arange(1, dtype='>f8')
        La = np.arange(1, dtype='<f8')
        assert_equal((Ba + Ba).dtype, np.dtype('f8'))
        assert_equal((Ba + La).dtype, np.dtype('f8'))
        assert_equal((La + Ba).dtype, np.dtype('f8'))
        assert_equal((La + La).dtype, np.dtype('f8'))
        assert_equal(np.absolute(La).dtype, np.dtype('f8'))
        assert_equal(np.absolute(Ba).dtype, np.dtype('f8'))
        assert_equal(np.negative(La).dtype, np.dtype('f8'))
        assert_equal(np.negative(Ba).dtype, np.dtype('f8'))

    def test_incontiguous_array(self):
        msg = 'incontiguous memory layout of array'
        x = np.arange(64).reshape((2, 2, 2, 2, 2, 2))
        a = x[:, 0, :, 0, :, 0]
        b = x[:, 1, :, 1, :, 1]
        a[0, 0, 0] = -1
        msg2 = 'make sure it references to the original array'
        assert_equal(x[0, 0, 0, 0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)
        x = np.arange(24).reshape(2, 3, 4)
        a = x.T
        b = x.T
        a[0, 0, 0] = -1
        assert_equal(x[0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a, b), np.sum(a * b, axis=-1), err_msg=msg)

    def test_output_argument(self):
        msg = 'output argument'
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(4).reshape((2, 1, 2)) + 1
        c = np.zeros((2, 3), dtype='int')
        umt.inner1d(a, b, c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
        msg = 'output argument with type cast'
        c = np.zeros((2, 3), dtype='int16')
        umt.inner1d(a, b, c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c)
        assert_array_equal(c, np.sum(a * b, axis=-1), err_msg=msg)
        msg = 'output argument with incontiguous layout'
        c = np.zeros((2, 3, 4), dtype='int16')
        umt.inner1d(a, b, c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a * b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a * b, axis=-1), err_msg=msg)

    def test_axes_argument(self):
        inner1d = umt.inner1d
        a = np.arange(27.0).reshape((3, 3, 3))
        b = np.arange(10.0, 19.0).reshape((3, 1, 3))
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[(-1,), (-1,), ()])
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[-1, -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[(-1,), -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[-1, -1])
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[np.int8(-1), np.array(-1, dtype=np.int32)])
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[0, 0])
        assert_array_equal(c, (a * b).sum(0))
        c = inner1d(a, b, axes=[0, 2])
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        assert_raises(TypeError, inner1d, a, b, axes=-1)
        assert_raises(ValueError, inner1d, a, b, axes=[-1])
        assert_raises(TypeError, inner1d, a, b, axes=[-1.0, -1.0])
        assert_raises(TypeError, inner1d, a, b, axes=[(-1.0,), -1])
        assert_raises(TypeError, inner1d, a, b, axes=[None, 1])
        assert_raises(np.AxisError, inner1d, a, b, axes=[-1, -1, -1])
        assert_raises(np.AxisError, inner1d, a, b, axes=[-1, -1, (-1,)])
        assert_raises(np.AxisError, inner1d, a, b, axes=[-1, (-2, -1), ()])
        assert_raises(ValueError, inner1d, a, b, axes=[0, 1])
        mm = umt.matrix_multiply
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(8).reshape((2, 2, 2, 1)) + 1
        c = mm(a, b)
        assert_array_equal(c, np.matmul(a, b))
        c = mm(a, b, axes=[(-2, -1), (-2, -1), (-2, -1)])
        assert_array_equal(c, np.matmul(a, b))
        c = mm(a, b, axes=[(1, 2), (2, 3), (2, 3)])
        assert_array_equal(c, np.matmul(a, b))
        c = mm(a, b, axes=[(0, -1), (1, 2), (-2, -1)])
        assert_array_equal(c, np.matmul(a.transpose(1, 0, 2), b.transpose(0, 3, 1, 2)))
        c = np.empty((2, 2, 3, 1))
        d = mm(a, b, out=c, axes=[(1, 2), (2, 3), (2, 3)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b))
        c = np.empty((1, 2, 2, 3))
        d = mm(a, b, out=c, axes=[(-2, -1), (-2, -1), (3, 0)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b).transpose(3, 0, 1, 2))
        assert_raises(TypeError, mm, a, b, axis=1)
        assert_raises(TypeError, mm, a, b, axes=1)
        assert_raises(TypeError, mm, a, b, axes=((-2, -1), (-2, -1), (-2, -1)))
        assert_raises(ValueError, mm, a, b, axes=[])
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1)])
        assert_raises(TypeError, mm, a, b, axes=[None, None, None])
        assert_raises(TypeError, mm, a, b, axes=[[-2, -1], [-2, -1], [-2, -1]])
        assert_raises(TypeError, mm, a, b, axes=[(-2, -1), (-2, -1), [-2, -1]])
        assert_raises(TypeError, mm, a, b, axes=[(-2, -1), (-2, -1), None])
        assert_raises(np.AxisError, mm, a, b, axes=[-1, -1, -1])
        assert_raises(np.AxisError, mm, a, b, axes=[(-2, -1), (-2, -1), -1])
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1), (-2, -1), (-2, -2)])
        z = np.zeros((2, 2))
        assert_raises(ValueError, mm, z, z[0])
        assert_raises(ValueError, mm, z, z, out=z[:, 0])
        assert_raises(ValueError, mm, z[1], z, axes=[0, 1])
        assert_raises(ValueError, mm, z, z, out=z[0], axes=[0, 1])
        assert_raises(TypeError, np.add, 1.0, 1.0, axes=[0])
        assert_raises(TypeError, mm, z, z, axes=[0, 1], parrot=True)

    def test_axis_argument(self):
        inner1d = umt.inner1d
        a = np.arange(27.0).reshape((3, 3, 3))
        b = np.arange(10.0, 19.0).reshape((3, 1, 3))
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axis=-1)
        assert_array_equal(c, (a * b).sum(-1))
        out = np.zeros_like(c)
        d = inner1d(a, b, axis=-1, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        c = inner1d(a, b, axis=0)
        assert_array_equal(c, (a * b).sum(0))
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, axis=0), np.sum(a * b * w, axis=0))
        assert_array_equal(umt.cumsum(a, axis=0), np.cumsum(a, axis=0))
        assert_array_equal(umt.cumsum(a, axis=-1), np.cumsum(a, axis=-1))
        out = np.empty_like(a)
        b = umt.cumsum(a, out=out, axis=0)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=0))
        b = umt.cumsum(a, out=out, axis=1)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=-1))
        assert_raises(TypeError, inner1d, a, b, axis=0, axes=[0, 0])
        assert_raises(TypeError, inner1d, a, b, axis=[0])
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, axis=1)
        out = np.empty((1, 2, 3), dtype=a.dtype)
        assert_raises(ValueError, umt.cumsum, a, out=out, axis=0)
        assert_raises(TypeError, np.add, 1.0, 1.0, axis=0)

    def test_keepdims_argument(self):
        inner1d = umt.inner1d
        a = np.arange(27.0).reshape((3, 3, 3))
        b = np.arange(10.0, 19.0).reshape((3, 1, 3))
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        out = np.zeros_like(c)
        d = inner1d(a, b, keepdims=True, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        c = inner1d(a, b, axis=-1, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1, keepdims=False))
        c = inner1d(a, b, axis=-1, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axis=0, keepdims=False)
        assert_array_equal(c, (a * b).sum(0, keepdims=False))
        c = inner1d(a, b, axis=0, keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = inner1d(a, b, axes=[(-1,), (-1,), ()], keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[(-1,), (-1,), (-1,)], keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axes=[0, 0], keepdims=False)
        assert_array_equal(c, (a * b).sum(0))
        c = inner1d(a, b, axes=[0, 0, 0], keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = inner1d(a, b, axes=[0, 2], keepdims=False)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        c = inner1d(a, b, axes=[0, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axes=[0, 2, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axes=[0, 2, 0], keepdims=True)
        assert_array_equal(c, (a * b.transpose(2, 0, 1)).sum(0, keepdims=True))
        c = inner1d(a, b, axes=[0, 2, 1], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 0, 2) * b.transpose(0, 2, 1)).sum(1, keepdims=True))
        a = np.eye(3) * np.arange(4.0)[:, np.newaxis, np.newaxis]
        expected = uml.det(a)
        c = uml.det(a, keepdims=False)
        assert_array_equal(c, expected)
        c = uml.det(a, keepdims=True)
        assert_array_equal(c, expected[:, np.newaxis, np.newaxis])
        a = np.eye(3) * np.arange(4.0)[:, np.newaxis, np.newaxis]
        expected_s, expected_l = uml.slogdet(a)
        cs, cl = uml.slogdet(a, keepdims=False)
        assert_array_equal(cs, expected_s)
        assert_array_equal(cl, expected_l)
        cs, cl = uml.slogdet(a, keepdims=True)
        assert_array_equal(cs, expected_s[:, np.newaxis, np.newaxis])
        assert_array_equal(cl, expected_l[:, np.newaxis, np.newaxis])
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, keepdims=True), np.sum(a * b * w, axis=-1, keepdims=True))
        assert_array_equal(umt.innerwt(a, b, w, axis=0, keepdims=True), np.sum(a * b * w, axis=0, keepdims=True))
        assert_raises(TypeError, inner1d, a, b, keepdims='true')
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, keepdims=True)
        assert_raises(TypeError, mm, a, b, keepdims=False)
        assert_raises(TypeError, np.add, 1.0, 1.0, keepdims=False)

    def test_innerwt(self):
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))
        a = np.arange(100, 124).reshape((2, 3, 4))
        b = np.arange(200, 224).reshape((2, 3, 4))
        w = np.arange(300, 324).reshape((2, 3, 4))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))

    def test_innerwt_empty(self):
        """Test generalized ufunc with zero-sized operands"""
        a = np.array([], dtype='f8')
        b = np.array([], dtype='f8')
        w = np.array([], dtype='f8')
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a * b * w, axis=-1))

    def test_cross1d(self):
        """Test with fixed-sized signature."""
        a = np.eye(3)
        assert_array_equal(umt.cross1d(a, a), np.zeros((3, 3)))
        out = np.zeros((3, 3))
        result = umt.cross1d(a[0], a, out)
        assert_(result is out)
        assert_array_equal(result, np.vstack((np.zeros(3), a[2], -a[1])))
        assert_raises(ValueError, umt.cross1d, np.eye(4), np.eye(4))
        assert_raises(ValueError, umt.cross1d, a, np.arange(4.0))
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.0), np.zeros((3, 4)))
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.0), np.zeros(3))

    def test_can_ignore_signature(self):
        mat = np.arange(12).reshape((2, 3, 2))
        single_vec = np.arange(2)
        col_vec = single_vec[:, np.newaxis]
        col_vec_array = np.arange(8).reshape((2, 2, 2, 1)) + 1
        mm_col_vec = umt.matrix_multiply(mat, col_vec)
        matmul_col_vec = umt.matmul(mat, col_vec)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        assert_raises(ValueError, umt.matrix_multiply, mat, single_vec)
        matmul_col = umt.matmul(mat, single_vec)
        assert_array_equal(matmul_col, mm_col_vec.squeeze())
        mm_col_vec = umt.matrix_multiply(mat, col_vec_array)
        matmul_col_vec = umt.matmul(mat, col_vec_array)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        single_vec = np.arange(3)
        row_vec = single_vec[np.newaxis, :]
        row_vec_array = np.arange(24).reshape((4, 2, 1, 1, 3)) + 1
        mm_row_vec = umt.matrix_multiply(row_vec, mat)
        matmul_row_vec = umt.matmul(row_vec, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        assert_raises(ValueError, umt.matrix_multiply, single_vec, mat)
        matmul_row = umt.matmul(single_vec, mat)
        assert_array_equal(matmul_row, mm_row_vec.squeeze())
        mm_row_vec = umt.matrix_multiply(row_vec_array, mat)
        matmul_row_vec = umt.matmul(row_vec_array, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        col_vec = row_vec.T
        col_vec_array = row_vec_array.swapaxes(-2, -1)
        mm_row_col_vec = umt.matrix_multiply(row_vec, col_vec)
        matmul_row_col_vec = umt.matmul(row_vec, col_vec)
        assert_array_equal(matmul_row_col_vec, mm_row_col_vec)
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec)
        matmul_row_col = umt.matmul(single_vec, single_vec)
        assert_array_equal(matmul_row_col, mm_row_col_vec.squeeze())
        mm_row_col_array = umt.matrix_multiply(row_vec_array, col_vec_array)
        matmul_row_col_array = umt.matmul(row_vec_array, col_vec_array)
        assert_array_equal(matmul_row_col_array, mm_row_col_array)
        out = np.zeros_like(mm_row_col_array)
        out = umt.matrix_multiply(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        out[:] = 0
        out = umt.matmul(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        out = np.zeros_like(mm_row_col_vec)
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec, out)
        out = umt.matmul(single_vec, single_vec, out)
        assert_array_equal(out, mm_row_col_vec.squeeze())

    def test_matrix_multiply(self):
        self.compare_matrix_multiply_results(np.int64)
        self.compare_matrix_multiply_results(np.double)

    def test_matrix_multiply_umath_empty(self):
        res = umt.matrix_multiply(np.ones((0, 10)), np.ones((10, 0)))
        assert_array_equal(res, np.zeros((0, 0)))
        res = umt.matrix_multiply(np.ones((10, 0)), np.ones((0, 10)))
        assert_array_equal(res, np.zeros((10, 10)))

    def compare_matrix_multiply_results(self, tp):
        d1 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        d2 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        msg = 'matrix multiply on type %s' % d1.dtype.name

        def permute_n(n):
            if n == 1:
                return ([0],)
            ret = ()
            base = permute_n(n - 1)
            for perm in base:
                for i in range(n):
                    new = perm + [n - 1]
                    new[n - 1] = new[i]
                    new[i] = n - 1
                    ret += (new,)
            return ret

        def slice_n(n):
            if n == 0:
                return ((),)
            ret = ()
            base = slice_n(n - 1)
            for sl in base:
                ret += (sl + (slice(None),),)
                ret += (sl + (slice(0, 1),),)
            return ret

        def broadcastable(s1, s2):
            return s1 == s2 or s1 == 1 or s2 == 1
        permute_3 = permute_n(3)
        slice_3 = slice_n(3) + ((slice(None, None, -1),) * 3,)
        ref = True
        for p1 in permute_3:
            for p2 in permute_3:
                for s1 in slice_3:
                    for s2 in slice_3:
                        a1 = d1.transpose(p1)[s1]
                        a2 = d2.transpose(p2)[s2]
                        ref = ref and a1.base is not None
                        ref = ref and a2.base is not None
                        if a1.shape[-1] == a2.shape[-2] and broadcastable(a1.shape[0], a2.shape[0]):
                            assert_array_almost_equal(umt.matrix_multiply(a1, a2), np.sum(a2[..., np.newaxis].swapaxes(-3, -1) * a1[..., np.newaxis, :], axis=-1), err_msg=msg + ' %s %s' % (str(a1.shape), str(a2.shape)))
        assert_equal(ref, True, err_msg='reference check')

    def test_euclidean_pdist(self):
        a = np.arange(12, dtype=float).reshape(4, 3)
        out = np.empty((a.shape[0] * (a.shape[0] - 1) // 2,), dtype=a.dtype)
        umt.euclidean_pdist(a, out)
        b = np.sqrt(np.sum((a[:, None] - a) ** 2, axis=-1))
        b = b[~np.tri(a.shape[0], dtype=bool)]
        assert_almost_equal(out, b)
        assert_raises(ValueError, umt.euclidean_pdist, a)

    def test_cumsum(self):
        a = np.arange(10)
        result = umt.cumsum(a)
        assert_array_equal(result, a.cumsum())

    def test_object_logical(self):
        a = np.array([3, None, True, False, 'test', ''], dtype=object)
        assert_equal(np.logical_or(a, None), np.array([x or None for x in a], dtype=object))
        assert_equal(np.logical_or(a, True), np.array([x or True for x in a], dtype=object))
        assert_equal(np.logical_or(a, 12), np.array([x or 12 for x in a], dtype=object))
        assert_equal(np.logical_or(a, 'blah'), np.array([x or 'blah' for x in a], dtype=object))
        assert_equal(np.logical_and(a, None), np.array([x and None for x in a], dtype=object))
        assert_equal(np.logical_and(a, True), np.array([x and True for x in a], dtype=object))
        assert_equal(np.logical_and(a, 12), np.array([x and 12 for x in a], dtype=object))
        assert_equal(np.logical_and(a, 'blah'), np.array([x and 'blah' for x in a], dtype=object))
        assert_equal(np.logical_not(a), np.array([not x for x in a], dtype=object))
        assert_equal(np.logical_or.reduce(a), 3)
        assert_equal(np.logical_and.reduce(a), None)

    def test_object_comparison(self):

        class HasComparisons:

            def __eq__(self, other):
                return '=='
        arr0d = np.array(HasComparisons())
        assert_equal(arr0d == arr0d, True)
        assert_equal(np.equal(arr0d, arr0d), True)
        arr1d = np.array([HasComparisons()])
        assert_equal(arr1d == arr1d, np.array([True]))
        assert_equal(np.equal(arr1d, arr1d), np.array([True]))
        assert_equal(np.equal(arr1d, arr1d, dtype=object), np.array(['==']))

    def test_object_array_reduction(self):
        a = np.array(['a', 'b', 'c'], dtype=object)
        assert_equal(np.sum(a), 'abc')
        assert_equal(np.max(a), 'c')
        assert_equal(np.min(a), 'a')
        a = np.array([True, False, True], dtype=object)
        assert_equal(np.sum(a), 2)
        assert_equal(np.prod(a), 0)
        assert_equal(np.any(a), True)
        assert_equal(np.all(a), False)
        assert_equal(np.max(a), True)
        assert_equal(np.min(a), False)
        assert_equal(np.array([[1]], dtype=object).sum(), 1)
        assert_equal(np.array([[[1, 2]]], dtype=object).sum((0, 1)), [1, 2])
        assert_equal(np.array([1], dtype=object).sum(initial=1), 2)
        assert_equal(np.array([[1], [2, 3]], dtype=object).sum(initial=[0], where=[False, True]), [0, 2, 3])

    def test_object_array_accumulate_inplace(self):
        arr = np.ones(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        np.add.accumulate(arr, out=arr)
        np.add.accumulate(arr, out=arr)
        assert_array_equal(arr, np.array([[1] * i for i in [1, 3, 6, 10]], dtype=object))
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        np.add.accumulate(arr, out=arr, axis=-1)
        np.add.accumulate(arr, out=arr, axis=-1)
        assert_array_equal(arr[0, :], np.array([[2] * i for i in [1, 3, 6, 10]], dtype=object))

    def test_object_array_accumulate_failure(self):
        res = np.add.accumulate(np.array([1, 0, 2], dtype=object))
        assert_array_equal(res, np.array([1, 1, 3], dtype=object))
        with pytest.raises(TypeError):
            np.add.accumulate([1, None, 2])

    def test_object_array_reduceat_inplace(self):
        arr = np.empty(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        out = np.empty(4, dtype=object)
        out[:] = [[1] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr)
        np.add.reduceat(arr, np.arange(4), out=arr)
        assert_array_equal(arr, out)
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        out = np.ones((2, 4), dtype=object)
        out[0, :] = [[2] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        assert_array_equal(arr, out)

    def test_object_array_reduceat_failure(self):
        res = np.add.reduceat(np.array([1, None, 2], dtype=object), [1, 2])
        assert_array_equal(res, np.array([None, 2], dtype=object))
        with pytest.raises(TypeError):
            np.add.reduceat([1, None, 2], [0, 2])

    def test_zerosize_reduction(self):
        for a in [[], np.array([], dtype=object)]:
            assert_equal(np.sum(a), 0)
            assert_equal(np.prod(a), 1)
            assert_equal(np.any(a), False)
            assert_equal(np.all(a), True)
            assert_raises(ValueError, np.max, a)
            assert_raises(ValueError, np.min, a)

    def test_axis_out_of_bounds(self):
        a = np.array([False, False])
        assert_raises(np.AxisError, a.all, axis=1)
        a = np.array([False, False])
        assert_raises(np.AxisError, a.all, axis=-2)
        a = np.array([False, False])
        assert_raises(np.AxisError, a.any, axis=1)
        a = np.array([False, False])
        assert_raises(np.AxisError, a.any, axis=-2)

    def test_scalar_reduction(self):
        assert_equal(np.sum(3, axis=0), 3)
        assert_equal(np.prod(3.5, axis=0), 3.5)
        assert_equal(np.any(True, axis=0), True)
        assert_equal(np.all(False, axis=0), False)
        assert_equal(np.max(3, axis=0), 3)
        assert_equal(np.min(2.5, axis=0), 2.5)
        assert_equal(np.power.reduce(3), 3)
        assert_(type(np.prod(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.sum(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.max(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.min(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.any(0, axis=0)) is np.bool_)

        class MyArray(np.ndarray):
            pass
        a = np.array(1).view(MyArray)
        assert_(type(np.any(a)) is MyArray)

    def test_casting_out_param(self):
        a = np.ones((200, 100), np.int64)
        b = np.ones((200, 100), np.int64)
        c = np.ones((200, 100), np.float64)
        np.add(a, b, out=c)
        assert_equal(c, 2)
        a = np.zeros(65536)
        b = np.zeros(65536, dtype=np.float32)
        np.subtract(a, 0, out=b)
        assert_equal(b, 0)

    def test_where_param(self):
        a = np.arange(7)
        b = np.ones(7)
        c = np.zeros(7)
        np.add(a, b, out=c, where=a % 2 == 1)
        assert_equal(c, [0, 2, 0, 4, 0, 6, 0])
        a = np.arange(4).reshape(2, 2) + 2
        np.power(a, [2, 3], out=a, where=[[0, 1], [1, 0]])
        assert_equal(a, [[2, 27], [16, 5]])
        np.subtract(a, 2, out=a, where=[True, False])
        assert_equal(a, [[0, 27], [14, 5]])

    def test_where_param_buffer_output(self):
        a = np.ones(10, np.int64)
        b = np.ones(10, np.int64)
        c = 1.5 * np.ones(10, np.float64)
        np.add(a, b, out=c, where=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        assert_equal(c, [2, 1.5, 1.5, 2, 1.5, 1.5, 2, 2, 2, 1.5])

    def test_where_param_alloc(self):
        a = np.array([1], dtype=np.int64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])
        a = np.array([1], dtype=np.float64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])

    def test_where_with_broadcasting(self):
        a = np.random.random((5000, 4))
        b = np.random.random((5000, 1))
        where = a > 0.3
        out = np.full_like(a, 0)
        np.less(a, b, where=where, out=out)
        b_where = np.broadcast_to(b, a.shape)[where]
        assert_array_equal(a[where] < b_where, out[where].astype(bool))
        assert not out[~where].any()

    def check_identityless_reduction(self, a):
        a[...] = 1
        a[1, 0, 0] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [1, 0])
        assert_equal(np.minimum.reduce(a, axis=0), [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1), [[1, 1, 1, 1], [0, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2), [[1, 1, 1], [0, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)
        a[...] = 1
        a[0, 1, 0] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [1, 0, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0), [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1), [[0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2), [[1, 0, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)
        a[...] = 1
        a[0, 0, 1] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [1, 0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0), [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1), [[1, 0, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2), [[0, 1, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

    @requires_memory(6 * 1024 ** 3)
    @pytest.mark.skipif(sys.maxsize < 2 ** 32, reason='test array too large for 32bit platform')
    def test_identityless_reduction_huge_array(self):
        arr = np.zeros((2, 2 ** 31), 'uint8')
        arr[:, 0] = [1, 3]
        arr[:, -1] = [4, 1]
        res = np.maximum.reduce(arr, axis=0)
        del arr
        assert res[0] == 3
        assert res[-1] == 4

    def test_identityless_reduction_corder(self):
        a = np.empty((2, 3, 4), order='C')
        self.check_identityless_reduction(a)

    def test_identityless_reduction_forder(self):
        a = np.empty((2, 3, 4), order='F')
        self.check_identityless_reduction(a)

    def test_identityless_reduction_otherorder(self):
        a = np.empty((2, 4, 3), order='C').swapaxes(1, 2)
        self.check_identityless_reduction(a)

    def test_identityless_reduction_noncontig(self):
        a = np.empty((3, 5, 4), order='C').swapaxes(1, 2)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_identityless_reduction_noncontig_unaligned(self):
        a = np.empty((3 * 4 * 5 * 8 + 1,), dtype='i1')
        a = a[1:].view(dtype='f8')
        a.shape = (3, 4, 5)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_reduce_identity_depends_on_loop(self):
        """
        The type of the result should always depend on the selected loop, not
        necessarily the output (only relevant for object arrays).
        """
        assert type(np.add.reduce([], dtype=object)) is int
        out = np.array(None, dtype=object)
        np.add.reduce([], out=out, dtype=np.float64)
        assert type(out[()]) is float

    def test_initial_reduction(self):
        assert_equal(np.maximum.reduce([], initial=0), 0)
        assert_equal(np.minimum.reduce([], initial=np.inf), np.inf)
        assert_equal(np.maximum.reduce([], initial=-np.inf), -np.inf)
        assert_equal(np.minimum.reduce([5], initial=4), 4)
        assert_equal(np.maximum.reduce([4], initial=5), 5)
        assert_equal(np.maximum.reduce([5], initial=4), 5)
        assert_equal(np.minimum.reduce([4], initial=5), 4)
        assert_raises(ValueError, np.minimum.reduce, [], initial=None)
        assert_raises(ValueError, np.add.reduce, [], initial=None)
        with pytest.raises(ValueError):
            np.add.reduce([], initial=None, dtype=object)
        assert_equal(np.add.reduce([], initial=np._NoValue), 0)
        a = np.array([10], dtype=object)
        res = np.add.reduce(a, initial=5)
        assert_equal(res, 15)

    def test_empty_reduction_and_idenity(self):
        arr = np.zeros((0, 5))
        assert np.true_divide.reduce(arr, axis=1).shape == (0,)
        with pytest.raises(ValueError):
            np.true_divide.reduce(arr, axis=0)
        arr = np.zeros((0, 0, 5))
        with pytest.raises(ValueError):
            np.true_divide.reduce(arr, axis=1)
        res = np.true_divide.reduce(arr, axis=1, initial=1)
        assert_array_equal(res, np.ones((0, 5)))

    @pytest.mark.parametrize('axis', (0, 1, None))
    @pytest.mark.parametrize('where', (np.array([False, True, True]), np.array([[True], [False], [True]]), np.array([[True, False, False], [False, True, False], [False, True, True]])))
    def test_reduction_with_where(self, axis, where):
        a = np.arange(9.0).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.zeros_like(a)
        np.positive(a, out=a_check, where=where)
        res = np.add.reduce(a, axis=axis, where=where)
        check = a_check.sum(axis)
        assert_equal(res, check)
        assert_array_equal(a, a_copy)

    @pytest.mark.parametrize(('axis', 'where'), ((0, np.array([True, False, True])), (1, [True, True, False]), (None, True)))
    @pytest.mark.parametrize('initial', (-np.inf, 5.0))
    def test_reduction_with_where_and_initial(self, axis, where, initial):
        a = np.arange(9.0).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.full(a.shape, -np.inf)
        np.positive(a, out=a_check, where=where)
        res = np.maximum.reduce(a, axis=axis, where=where, initial=initial)
        check = a_check.max(axis, initial=initial)
        assert_equal(res, check)

    def test_reduction_where_initial_needed(self):
        a = np.arange(9.0).reshape(3, 3)
        m = [False, True, False]
        assert_raises(ValueError, np.maximum.reduce, a, where=m)

    def test_identityless_reduction_nonreorderable(self):
        a = np.array([[8.0, 2.0, 2.0], [1.0, 0.5, 0.25]])
        res = np.divide.reduce(a, axis=0)
        assert_equal(res, [8.0, 4.0, 8.0])
        res = np.divide.reduce(a, axis=1)
        assert_equal(res, [2.0, 8.0])
        res = np.divide.reduce(a, axis=())
        assert_equal(res, a)
        assert_raises(ValueError, np.divide.reduce, a, axis=(0, 1))

    def test_reduce_zero_axis(self):

        def ok(f, *args, **kwargs):
            f(*args, **kwargs)

        def err(f, *args, **kwargs):
            assert_raises(ValueError, f, *args, **kwargs)

        def t(expect, func, n, m):
            expect(func, np.zeros((n, m)), axis=1)
            expect(func, np.zeros((m, n)), axis=0)
            expect(func, np.zeros((n // 2, n // 2, m)), axis=2)
            expect(func, np.zeros((n // 2, m, n // 2)), axis=1)
            expect(func, np.zeros((n, m // 2, m // 2)), axis=(1, 2))
            expect(func, np.zeros((m // 2, n, m // 2)), axis=(0, 2))
            expect(func, np.zeros((m // 3, m // 3, m // 3, n // 2, n // 2)), axis=(0, 1, 2))
            expect(func, np.zeros((10, m, n)), axis=(0, 1))
            expect(func, np.zeros((10, n, m)), axis=(0, 2))
            expect(func, np.zeros((m, 10, n)), axis=0)
            expect(func, np.zeros((10, m, n)), axis=1)
            expect(func, np.zeros((10, n, m)), axis=2)
        assert_equal(np.maximum.identity, None)
        t(ok, np.maximum.reduce, 30, 30)
        t(ok, np.maximum.reduce, 0, 30)
        t(err, np.maximum.reduce, 30, 0)
        t(err, np.maximum.reduce, 0, 0)
        err(np.maximum.reduce, [])
        np.maximum.reduce(np.zeros((0, 0)), axis=())
        t(ok, np.add.reduce, 30, 30)
        t(ok, np.add.reduce, 0, 30)
        t(ok, np.add.reduce, 30, 0)
        t(ok, np.add.reduce, 0, 0)
        np.add.reduce([])
        np.add.reduce(np.zeros((0, 0)), axis=())
        for uf in (np.maximum, np.add):
            uf.accumulate(np.zeros((30, 0)), axis=0)
            uf.accumulate(np.zeros((0, 30)), axis=0)
            uf.accumulate(np.zeros((30, 30)), axis=0)
            uf.accumulate(np.zeros((0, 0)), axis=0)

    def test_safe_casting(self):
        a = np.array([1, 2, 3], dtype=int)
        assert_array_equal(assert_no_warnings(np.add, a, 1.1), [2.1, 3.1, 4.1])
        assert_raises(TypeError, np.add, a, 1.1, out=a)

        def add_inplace(a, b):
            a += b
        assert_raises(TypeError, add_inplace, a, 1.1)
        assert_no_warnings(np.add, a, 1.1, out=a, casting='unsafe')
        assert_array_equal(a, [2, 3, 4])

    def test_ufunc_custom_out(self):
        a = np.array([0, 1, 2], dtype='i8')
        b = np.array([0, 1, 2], dtype='i8')
        c = np.empty(3, dtype=_rational_tests.rational)
        result = _rational_tests.test_add(a, b, c)
        target = np.array([0, 2, 4], dtype=_rational_tests.rational)
        assert_equal(result, target)
        result = _rational_tests.test_add(a, b)
        assert_equal(result, target)
        result = _rational_tests.test_add(a, b.astype(np.uint16), out=c)
        assert_equal(result, target)
        with assert_raises(TypeError):
            _rational_tests.test_add(a, np.uint16(2))

    def test_operand_flags(self):
        a = np.arange(16, dtype='l').reshape(4, 4)
        b = np.arange(9, dtype='l').reshape(3, 3)
        opflag_tests.inplace_add(a[:-1, :-1], b)
        assert_equal(a, np.array([[0, 2, 4, 3], [7, 9, 11, 7], [14, 16, 18, 11], [12, 13, 14, 15]], dtype='l'))
        a = np.array(0)
        opflag_tests.inplace_add(a, 3)
        assert_equal(a, 3)
        opflag_tests.inplace_add(a, [3, 4])
        assert_equal(a, 10)

    def test_struct_ufunc(self):
        import numpy.core._struct_ufunc_tests as struct_ufunc
        a = np.array([(1, 2, 3)], dtype='u8,u8,u8')
        b = np.array([(1, 2, 3)], dtype='u8,u8,u8')
        result = struct_ufunc.add_triplet(a, b)
        assert_equal(result, np.array([(2, 4, 6)], dtype='u8,u8,u8'))
        assert_raises(RuntimeError, struct_ufunc.register_fail)

    def test_custom_ufunc(self):
        a = np.array([_rational_tests.rational(1, 2), _rational_tests.rational(1, 3), _rational_tests.rational(1, 4)], dtype=_rational_tests.rational)
        b = np.array([_rational_tests.rational(1, 2), _rational_tests.rational(1, 3), _rational_tests.rational(1, 4)], dtype=_rational_tests.rational)
        result = _rational_tests.test_add_rationals(a, b)
        expected = np.array([_rational_tests.rational(1), _rational_tests.rational(2, 3), _rational_tests.rational(1, 2)], dtype=_rational_tests.rational)
        assert_equal(result, expected)

    def test_custom_ufunc_forced_sig(self):
        with assert_raises(TypeError):
            np.multiply(_rational_tests.rational(1), 1, signature=(_rational_tests.rational, int, None))

    def test_custom_array_like(self):

        class MyThing:
            __array_priority__ = 1000
            rmul_count = 0
            getitem_count = 0

            def __init__(self, shape):
                self.shape = shape

            def __len__(self):
                return self.shape[0]

            def __getitem__(self, i):
                MyThing.getitem_count += 1
                if not isinstance(i, tuple):
                    i = (i,)
                if len(i) > self.ndim:
                    raise IndexError('boo')
                return MyThing(self.shape[len(i):])

            def __rmul__(self, other):
                MyThing.rmul_count += 1
                return self
        np.float64(5) * MyThing((3, 3))
        assert_(MyThing.rmul_count == 1, MyThing.rmul_count)
        assert_(MyThing.getitem_count <= 2, MyThing.getitem_count)

    @pytest.mark.parametrize('a', (np.arange(10, dtype=int), np.arange(10, dtype=_rational_tests.rational)))
    def test_ufunc_at_basic(self, a):
        aa = a.copy()
        np.add.at(aa, [2, 5, 2], 1)
        assert_equal(aa, [0, 1, 4, 3, 4, 6, 6, 7, 8, 9])
        with pytest.raises(ValueError):
            np.add.at(aa, [2, 5, 3])
        aa = a.copy()
        np.negative.at(aa, [2, 5, 3])
        assert_equal(aa, [0, 1, -2, -3, 4, -5, 6, 7, 8, 9])
        aa = a.copy()
        b = np.array([100, 100, 100])
        np.add.at(aa, [2, 5, 2], b)
        assert_equal(aa, [0, 1, 202, 3, 4, 105, 6, 7, 8, 9])
        with pytest.raises(ValueError):
            np.negative.at(a, [2, 5, 3], [1, 2, 3])
        with pytest.raises(ValueError):
            np.add.at(a, [2, 5, 3], [[1, 2], 1])
    indexed_ufuncs = [np.add, np.subtract, np.multiply, np.floor_divide, np.maximum, np.minimum, np.fmax, np.fmin]

    @pytest.mark.parametrize('typecode', np.typecodes['AllInteger'] + np.typecodes['Float'])
    @pytest.mark.parametrize('ufunc', indexed_ufuncs)
    def test_ufunc_at_inner_loops(self, typecode, ufunc):
        if ufunc is np.divide and typecode in np.typecodes['AllInteger']:
            a = np.ones(100, dtype=typecode)
            indx = np.random.randint(100, size=30, dtype=np.intp)
            vals = np.arange(1, 31, dtype=typecode)
        else:
            a = np.ones(1000, dtype=typecode)
            indx = np.random.randint(1000, size=3000, dtype=np.intp)
            vals = np.arange(3000, dtype=typecode)
        atag = a.copy()
        with warnings.catch_warnings(record=True) as w_at:
            warnings.simplefilter('always')
            ufunc.at(a, indx, vals)
        with warnings.catch_warnings(record=True) as w_loop:
            warnings.simplefilter('always')
            for i, v in zip(indx, vals):
                ufunc(atag[i], v, out=atag[i:i + 1], casting='unsafe')
        assert_equal(atag, a)
        if len(w_loop) > 0:
            assert len(w_at) > 0
            assert w_at[0].category == w_loop[0].category
            assert str(w_at[0].message)[:10] == str(w_loop[0].message)[:10]

    @pytest.mark.parametrize('typecode', np.typecodes['Complex'])
    @pytest.mark.parametrize('ufunc', [np.add, np.subtract, np.multiply])
    def test_ufunc_at_inner_loops_complex(self, typecode, ufunc):
        a = np.ones(10, dtype=typecode)
        indx = np.concatenate([np.ones(6, dtype=np.intp), np.full(18, 4, dtype=np.intp)])
        value = a.dtype.type(1j)
        ufunc.at(a, indx, value)
        expected = np.ones_like(a)
        if ufunc is np.multiply:
            expected[1] = expected[4] = -1
        else:
            expected[1] += 6 * (value if ufunc is np.add else -value)
            expected[4] += 18 * (value if ufunc is np.add else -value)
        assert_array_equal(a, expected)

    def test_ufunc_at_ellipsis(self):
        arr = np.zeros(5)
        np.add.at(arr, slice(None), np.ones(5))
        assert_array_equal(arr, np.ones(5))

    def test_ufunc_at_negative(self):
        arr = np.ones(5, dtype=np.int32)
        indx = np.arange(5)
        umt.indexed_negative.at(arr, indx)
        assert np.all(arr == [-1, -1, -1, -200, -1])

    def test_ufunc_at_large(self):
        indices = np.zeros(8195, dtype=np.int16)
        b = np.zeros(8195, dtype=float)
        b[0] = 10
        b[1] = 5
        b[8192:] = 100
        a = np.zeros(1, dtype=float)
        np.add.at(a, indices, b)
        assert a[0] == b.sum()

    def test_cast_index_fastpath(self):
        arr = np.zeros(10)
        values = np.ones(100000)
        index = np.zeros(len(values), dtype=np.uint8)
        np.add.at(arr, index, values)
        assert arr[0] == len(values)

    @pytest.mark.parametrize('value', [np.ones(1), np.ones(()), np.float64(1.0), 1.0])
    def test_ufunc_at_scalar_value_fastpath(self, value):
        arr = np.zeros(1000)
        index = np.repeat(np.arange(1000), 2)
        np.add.at(arr, index, value)
        assert_array_equal(arr, np.full_like(arr, 2 * value))

    def test_ufunc_at_multiD(self):
        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, (slice(None), [1, 2, 1]), b)
        assert_equal(a, [[0, 201, 102], [3, 404, 205], [6, 607, 308]])
        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), [1, 2, 1]), b)
        assert_equal(a, [[[0, 401, 202], [3, 404, 205], [6, 407, 208]], [[9, 410, 211], [12, 413, 214], [15, 416, 217]], [[18, 419, 220], [21, 422, 223], [24, 425, 226]]])
        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, ([1, 2, 1], slice(None)), b)
        assert_equal(a, [[0, 1, 2], [403, 404, 405], [206, 207, 208]])
        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), [1, 2, 1], slice(None)), b)
        assert_equal(a, [[[0, 1, 2], [203, 404, 605], [106, 207, 308]], [[9, 10, 11], [212, 413, 614], [115, 216, 317]], [[18, 19, 20], [221, 422, 623], [124, 225, 326]]])
        a = np.arange(9).reshape(3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (0, [1, 2, 1]), b)
        assert_equal(a, [[0, 401, 202], [3, 4, 5], [6, 7, 8]])
        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, ([1, 2, 1], 0, slice(None)), b)
        assert_equal(a, [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[209, 410, 611], [12, 13, 14], [15, 16, 17]], [[118, 219, 320], [21, 22, 23], [24, 25, 26]]])
        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), slice(None)), b)
        assert_equal(a, [[[100, 201, 302], [103, 204, 305], [106, 207, 308]], [[109, 210, 311], [112, 213, 314], [115, 216, 317]], [[118, 219, 320], [121, 222, 323], [124, 225, 326]]])

    def test_ufunc_at_0D(self):
        a = np.array(0)
        np.add.at(a, (), 1)
        assert_equal(a, 1)
        assert_raises(IndexError, np.add.at, a, 0, 1)
        assert_raises(IndexError, np.add.at, a, [], 1)

    def test_ufunc_at_dtypes(self):
        a = np.arange(10)
        np.power.at(a, [1, 2, 3, 2], 3.5)
        assert_equal(a, np.array([0, 1, 4414, 46, 4, 5, 6, 7, 8, 9]))

    def test_ufunc_at_boolean(self):
        a = np.arange(10)
        index = a % 2 == 0
        np.equal.at(a, index, [0, 2, 4, 6, 8])
        assert_equal(a, [1, 1, 1, 3, 1, 5, 1, 7, 1, 9])
        a = np.arange(10, dtype='u4')
        np.invert.at(a, [2, 5, 2])
        assert_equal(a, [0, 1, 2, 3, 4, 5 ^ 4294967295, 6, 7, 8, 9])

    def test_ufunc_at_advanced(self):
        orig = np.arange(4)
        a = orig[:, None][:, 0:0]
        np.add.at(a, [0, 1], 3)
        assert_array_equal(orig, np.arange(4))
        index = np.array([1, 2, 1], np.dtype('i').newbyteorder())
        values = np.array([1, 2, 3, 4], np.dtype('f').newbyteorder())
        np.add.at(values, index, 3)
        assert_array_equal(values, [1, 8, 6, 4])
        values = np.array(['a', 1], dtype=object)
        assert_raises(TypeError, np.add.at, values, [0, 1], 1)
        assert_array_equal(values, np.array(['a', 1], dtype=object))
        assert_raises(ValueError, np.modf.at, np.arange(10), [1])
        a = np.array([1, 2, 3])
        np.maximum.at(a, [0], 0)
        assert_equal(a, np.array([1, 2, 3]))

    @pytest.mark.parametrize('dtype', np.typecodes['AllInteger'] + np.typecodes['Float'])
    @pytest.mark.parametrize('ufunc', [np.add, np.subtract, np.divide, np.minimum, np.maximum])
    def test_at_negative_indexes(self, dtype, ufunc):
        a = np.arange(0, 10).astype(dtype)
        indxs = np.array([-1, 1, -1, 2]).astype(np.intp)
        vals = np.array([1, 5, 2, 10], dtype=a.dtype)
        expected = a.copy()
        for i, v in zip(indxs, vals):
            expected[i] = ufunc(expected[i], v)
        ufunc.at(a, indxs, vals)
        assert_array_equal(a, expected)
        assert np.all(indxs == [-1, 1, -1, 2])

    def test_at_not_none_signature(self):
        a = np.ones((2, 2, 2))
        b = np.ones((1, 2, 2))
        assert_raises(TypeError, np.matmul.at, a, [0], b)
        a = np.array([[[1, 2], [3, 4]]])
        assert_raises(TypeError, np.linalg._umath_linalg.det.at, a, [0])

    def test_at_no_loop_for_op(self):
        arr = np.ones(10, dtype=str)
        with pytest.raises(np.core._exceptions._UFuncNoLoopError):
            np.add.at(arr, [0, 1], [0, 1])

    def test_at_output_casting(self):
        arr = np.array([-1])
        np.equal.at(arr, [0], [0])
        assert arr[0] == 0

    def test_at_broadcast_failure(self):
        arr = np.arange(5)
        with pytest.raises(ValueError):
            np.add.at(arr, [0, 1], [1, 2, 3])

    def test_reduce_arguments(self):
        f = np.add.reduce
        d = np.ones((5, 2), dtype=int)
        o = np.ones((2,), dtype=d.dtype)
        r = o * 5
        assert_equal(f(d), r)
        assert_equal(f(d, axis=0), r)
        assert_equal(f(d, 0), r)
        assert_equal(f(d, 0, dtype=None), r)
        assert_equal(f(d, 0, dtype='i'), r)
        assert_equal(f(d, 0, 'i'), r)
        assert_equal(f(d, 0, None), r)
        assert_equal(f(d, 0, None, out=None), r)
        assert_equal(f(d, 0, None, out=o), r)
        assert_equal(f(d, 0, None, o), r)
        assert_equal(f(d, 0, None, None), r)
        assert_equal(f(d, 0, None, None, keepdims=False), r)
        assert_equal(f(d, 0, None, None, True), r.reshape((1,) + r.shape))
        assert_equal(f(d, 0, None, None, False, 0), r)
        assert_equal(f(d, 0, None, None, False, initial=0), r)
        assert_equal(f(d, 0, None, None, False, 0, True), r)
        assert_equal(f(d, 0, None, None, False, 0, where=True), r)
        assert_equal(f(d, axis=0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False, initial=0, where=True), r)
        assert_raises(TypeError, f)
        assert_raises(TypeError, f, d, 0, None, None, False, 0, True, 1)
        assert_raises(TypeError, f, d, 'invalid')
        assert_raises(TypeError, f, d, axis='invalid')
        assert_raises(TypeError, f, d, axis='invalid', dtype=None, keepdims=True)
        assert_raises(TypeError, f, d, 0, 'invalid')
        assert_raises(TypeError, f, d, dtype='invalid')
        assert_raises(TypeError, f, d, dtype='invalid', out=None)
        assert_raises(TypeError, f, d, 0, None, 'invalid')
        assert_raises(TypeError, f, d, out='invalid')
        assert_raises(TypeError, f, d, out='invalid', dtype=None)
        assert_raises(TypeError, f, d, 0, keepdims='invalid', dtype='invalid', out=None)
        assert_raises(TypeError, f, d, axis=0, dtype=None, invalid=0)
        assert_raises(TypeError, f, d, invalid=0)
        assert_raises(TypeError, f, d, 0, keepdims=True, invalid='invalid', out=None)
        assert_raises(TypeError, f, d, axis=0, dtype=None, keepdims=True, out=None, invalid=0)
        assert_raises(TypeError, f, d, axis=0, dtype=None, out=None, invalid=0)

    def test_structured_equal(self):

        class MyA(np.ndarray):

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return getattr(ufunc, method)(*(input.view(np.ndarray) for input in inputs), **kwargs)
        a = np.arange(12.0).reshape(4, 3)
        ra = a.view(dtype='f8,f8,f8').squeeze()
        mra = ra.view(MyA)
        target = np.array([True, False, False, False], dtype=bool)
        assert_equal(np.all(target == (mra == ra[0])), True)

    def test_scalar_equal(self):
        a = np.array(0.0)
        b = np.array('a')
        assert_(a != b)
        assert_(b != a)
        assert_(not a == b)
        assert_(not b == a)

    def test_NotImplemented_not_returned(self):
        binary_funcs = [np.power, np.add, np.subtract, np.multiply, np.divide, np.true_divide, np.floor_divide, np.bitwise_and, np.bitwise_or, np.bitwise_xor, np.left_shift, np.right_shift, np.fmax, np.fmin, np.fmod, np.hypot, np.logaddexp, np.logaddexp2, np.maximum, np.minimum, np.mod, np.greater, np.greater_equal, np.less, np.less_equal, np.equal, np.not_equal]
        a = np.array('1')
        b = 1
        c = np.array([1.0, 2.0])
        for f in binary_funcs:
            assert_raises(TypeError, f, a, b)
            assert_raises(TypeError, f, c, a)

    @pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or])
    @pytest.mark.parametrize('signature', [(None, None, object), (object, None, None), (None, object, None)])
    def test_logical_ufuncs_object_signatures(self, ufunc, signature):
        a = np.array([True, None, False], dtype=object)
        res = ufunc(a, a, signature=signature)
        assert res.dtype == object

    @pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
    @pytest.mark.parametrize('signature', [(bool, None, object), (object, None, bool), (None, object, bool)])
    def test_logical_ufuncs_mixed_object_signatures(self, ufunc, signature):
        a = np.array([True, None, False])
        with pytest.raises(TypeError):
            ufunc(a, a, signature=signature)

    @pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_support_anything(self, ufunc):
        a = np.array(b'1', dtype='V3')
        c = np.array([1.0, 2.0])
        assert_array_equal(ufunc(a, c), ufunc([True, True], True))
        assert ufunc.reduce(a) == True
        out = np.zeros(2, dtype=np.int32)
        expected = ufunc([True, True], True).astype(out.dtype)
        assert_array_equal(ufunc(a, c, out=out), expected)
        out = np.zeros((), dtype=np.int32)
        assert ufunc.reduce(a, out=out) == True
        a = np.array([3], dtype='i')
        out = np.zeros((), dtype=a.dtype)
        assert ufunc.reduce(a, out=out) == 1

    @pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_reject_string(self, ufunc):
        """
        Logical ufuncs are normally well defined by working with the boolean
        equivalent, i.e. casting all inputs to bools should work.

        However, casting strings to bools is *currently* weird, because it
        actually uses `bool(int(str))`.  Thus we explicitly reject strings.
        This test should succeed (and can probably just be removed) as soon as
        string to bool casts are well defined in NumPy.
        """
        with pytest.raises(TypeError, match='contain a loop with signature'):
            ufunc(['1'], ['3'])
        with pytest.raises(TypeError, match='contain a loop with signature'):
            ufunc.reduce(['1', '2', '0'])

    @pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_out_cast_check(self, ufunc):
        a = np.array('1')
        c = np.array([1.0, 2.0])
        out = a.copy()
        with pytest.raises(TypeError):
            ufunc(a, c, out=out, casting='equiv')

    def test_reducelike_byteorder_resolution(self):
        arr_be = np.arange(10, dtype='>i8')
        arr_le = np.arange(10, dtype='<i8')
        assert np.add.reduce(arr_be) == np.add.reduce(arr_le)
        assert_array_equal(np.add.accumulate(arr_be), np.add.accumulate(arr_le))
        assert_array_equal(np.add.reduceat(arr_be, [1]), np.add.reduceat(arr_le, [1]))

    def test_reducelike_out_promotes(self):
        arr = np.ones(1000, dtype=np.uint8)
        out = np.zeros((), dtype=np.uint16)
        assert np.add.reduce(arr, out=out) == 1000
        arr[:10] = 2
        assert np.multiply.reduce(arr, out=out) == 2 ** 10
        arr = np.full(5, 2 ** 25 - 1, dtype=np.int64)
        res = np.zeros((), dtype=np.float32)
        single_res = np.zeros((), dtype=np.float32)
        np.multiply.reduce(arr, out=single_res, dtype=np.float32)
        assert single_res != res

    def test_reducelike_output_needs_identical_cast(self):
        arr = np.ones(20, dtype='f8')
        out = np.empty((), dtype=arr.dtype.newbyteorder())
        expected = np.add.reduce(arr)
        np.add.reduce(arr, out=out)
        assert_array_equal(expected, out)
        out = np.empty(2, dtype=arr.dtype.newbyteorder())
        expected = np.add.reduceat(arr, [0, 1])
        np.add.reduceat(arr, [0, 1], out=out)
        assert_array_equal(expected, out)
        out = np.empty(arr.shape, dtype=arr.dtype.newbyteorder())
        expected = np.add.accumulate(arr)
        np.add.accumulate(arr, out=out)
        assert_array_equal(expected, out)

    def test_reduce_noncontig_output(self):
        x = np.arange(7 * 13 * 8, dtype=np.int16).reshape(7, 13, 8)
        x = x[4:6, 1:11:6, 1:5].transpose(1, 2, 0)
        y_base = np.arange(4 * 4, dtype=np.int16).reshape(4, 4)
        y = y_base[::2, :]
        y_base_copy = y_base.copy()
        r0 = np.add.reduce(x, out=y.copy(), axis=2)
        r1 = np.add.reduce(x, out=y, axis=2)
        assert_equal(r0, r1)
        assert_equal(y_base[1, :], y_base_copy[1, :])
        assert_equal(y_base[3, :], y_base_copy[3, :])

    @pytest.mark.parametrize('with_cast', [True, False])
    def test_reduceat_and_accumulate_out_shape_mismatch(self, with_cast):
        arr = np.arange(5)
        out = np.arange(3)
        if with_cast:
            out = out.astype(np.float64)
        with pytest.raises(ValueError, match='(shape|size)'):
            np.add.reduceat(arr, [0, 3], out=out)
        with pytest.raises(ValueError, match='(shape|size)'):
            np.add.accumulate(arr, out=out)

    @pytest.mark.parametrize('out_shape', [(), (1,), (3,), (1, 1), (1, 3), (4, 3)])
    @pytest.mark.parametrize('keepdims', [True, False])
    @pytest.mark.parametrize('f_reduce', [np.add.reduce, np.minimum.reduce])
    def test_reduce_wrong_dimension_output(self, f_reduce, keepdims, out_shape):
        a = np.arange(12.0).reshape(4, 3)
        out = np.empty(out_shape, a.dtype)
        correct_out = f_reduce(a, axis=0, keepdims=keepdims)
        if out_shape != correct_out.shape:
            with assert_raises(ValueError):
                f_reduce(a, axis=0, out=out, keepdims=keepdims)
        else:
            check = f_reduce(a, axis=0, out=out, keepdims=keepdims)
            assert_(check is out)
            assert_array_equal(check, correct_out)

    def test_reduce_output_does_not_broadcast_input(self):
        a = np.ones((1, 10))
        out_correct = np.empty((1, 1))
        out_incorrect = np.empty((3, 1))
        np.add.reduce(a, axis=-1, out=out_correct, keepdims=True)
        np.add.reduce(a, axis=-1, out=out_correct[:, 0], keepdims=False)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect, keepdims=True)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect[:, 0], keepdims=False)

    def test_reduce_output_subclass_ok(self):

        class MyArr(np.ndarray):
            pass
        out = np.empty(())
        np.add.reduce(np.ones(5), out=out)
        out = out.view(MyArr)
        assert np.add.reduce(np.ones(5), out=out) is out
        assert type(np.add.reduce(out)) is MyArr

    def test_no_doc_string(self):
        assert_('\n' not in umt.inner1d_no_doc.__doc__)

    def test_invalid_args(self):
        exc = pytest.raises(TypeError, np.sqrt, None)
        assert exc.match('loop of ufunc does not support')

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_finite(self, nat):
        try:
            assert not np.isfinite(nat)
        except TypeError:
            pass

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_nan(self, nat):
        try:
            assert np.isnan(nat)
        except TypeError:
            pass

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_inf(self, nat):
        try:
            assert not np.isinf(nat)
        except TypeError:
            pass