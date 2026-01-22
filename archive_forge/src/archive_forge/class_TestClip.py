import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestClip:

    def setup_method(self):
        self.nr = 5
        self.nc = 3

    def fastclip(self, a, m, M, out=None, **kwargs):
        return a.clip(m, M, out=out, **kwargs)

    def clip(self, a, m, M, out=None):
        selector = np.less(a, m) + 2 * np.greater(a, M)
        return selector.choose((a, m, M), out=out)

    def _generate_data(self, n, m):
        return randn(n, m)

    def _generate_data_complex(self, n, m):
        return randn(n, m) + 1j * rand(n, m)

    def _generate_flt_data(self, n, m):
        return randn(n, m).astype(np.float32)

    def _neg_byteorder(self, a):
        a = np.asarray(a)
        if sys.byteorder == 'little':
            a = a.astype(a.dtype.newbyteorder('>'))
        else:
            a = a.astype(a.dtype.newbyteorder('<'))
        return a

    def _generate_non_native_data(self, n, m):
        data = randn(n, m)
        data = self._neg_byteorder(data)
        assert_(not data.dtype.isnative)
        return data

    def _generate_int_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int64)

    def _generate_int32_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int32)

    @pytest.mark.parametrize('dtype', '?bhilqpBHILQPefdgFDGO')
    def test_ones_pathological(self, dtype):
        arr = np.ones(10, dtype=dtype)
        expected = np.zeros(10, dtype=dtype)
        actual = np.clip(arr, 1, 0)
        if dtype == 'O':
            assert actual.tolist() == expected.tolist()
        else:
            assert_equal(actual, expected)

    def test_simple_double(self):
        a = self._generate_data(self.nr, self.nc)
        m = 0.1
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_int(self):
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(int)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_array_double(self):
        a = self._generate_data(self.nr, self.nc)
        m = np.zeros(a.shape)
        M = m + 0.5
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_nonnative(self):
        a = self._generate_non_native_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = self._neg_byteorder(0.6)
        assert_(not M.dtype.isnative)
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_simple_complex(self):
        a = 3 * self._generate_data_complex(self.nr, self.nc)
        m = -0.5
        M = 1.0
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)
        a = 3 * self._generate_data(self.nr, self.nc)
        m = -0.5 + 1j
        M = 1.0 + 2j
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_clip_complex(self):
        a = np.ones(10, dtype=complex)
        m = a.min()
        M = a.max()
        am = self.fastclip(a, m, None)
        aM = self.fastclip(a, None, M)
        assert_array_strict_equal(am, a)
        assert_array_strict_equal(aM, a)

    def test_clip_non_contig(self):
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags['F_CONTIGUOUS'])
        assert_(not a.flags['C_CONTIGUOUS'])
        ac = self.fastclip(a, -1.6, 1.7)
        act = self.clip(a, -1.6, 1.7)
        assert_array_strict_equal(ac, act)

    def test_simple_out(self):
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    @pytest.mark.parametrize('casting', [None, 'unsafe'])
    def test_simple_int32_inout(self, casting):
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        if casting is None:
            with pytest.raises(TypeError):
                self.fastclip(a, m, M, ac, casting=casting)
        else:
            self.fastclip(a, m, M, ac, casting=casting)
            self.clip(a, m, M, act)
            assert_array_strict_equal(ac, act)

    def test_simple_int64_out(self):
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int64_inout(self):
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.fastclip(a, m, M, out=ac, casting='unsafe')
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int32_out(self):
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.fastclip(a, m, M, out=ac, casting='unsafe')
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_inplace_01(self):
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_simple_inplace_02(self):
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_noncontig_inplace(self):
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags['F_CONTIGUOUS'])
        assert_(not a.flags['C_CONTIGUOUS'])
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_equal(a, ac)

    def test_type_cast_01(self):
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_02(self):
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(np.int32)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_03(self):
        a = self._generate_int32_data(self.nr, self.nc)
        m = -2
        M = 4
        ac = self.fastclip(a, np.float64(m), np.float64(M))
        act = self.clip(a, np.float64(m), np.float64(M))
        assert_array_strict_equal(ac, act)

    def test_type_cast_04(self):
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float32(-2)
        M = np.float32(4)
        act = self.fastclip(a, m, M)
        ac = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_05(self):
        a = self._generate_int_data(self.nr, self.nc)
        m = -0.5
        M = 1.0
        ac = self.fastclip(a, m * np.zeros(a.shape), M)
        act = self.clip(a, m * np.zeros(a.shape), M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_06(self):
        a = self._generate_data(self.nr, self.nc)
        m = 0.5
        m_s = self._neg_byteorder(m)
        M = 1.0
        act = self.clip(a, m_s, M)
        ac = self.fastclip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_07(self):
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.0
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        act = a_s.clip(m, M)
        ac = self.fastclip(a_s, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_08(self):
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 1.0
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        ac = self.fastclip(a_s, m, M)
        act = a_s.clip(m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_09(self):
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.0
        m_s = self._neg_byteorder(m)
        assert_(not m_s.dtype.isnative)
        ac = self.fastclip(a, m_s, M)
        act = self.clip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_10(self):
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.float32(-0.5)
        M = np.float32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_strict_equal(ac, act)

    def test_type_cast_11(self):
        a = self._generate_non_native_data(self.nr, self.nc)
        b = a.copy()
        b = b.astype(b.dtype.newbyteorder('>'))
        bt = b.copy()
        m = -0.5
        M = 1.0
        self.fastclip(a, m, M, out=b)
        self.clip(a, m, M, out=bt)
        assert_array_strict_equal(b, bt)

    def test_type_cast_12(self):
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.int32(0)
        M = np.int32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple(self):
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple2(self):
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.fastclip(a, m, M, out=ac, casting='unsafe')
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple_int32(self):
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_int32(self):
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.fastclip(a, m, M, out=ac, casting='unsafe')
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_outint32(self):
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.fastclip(a, m, M, out=ac, casting='unsafe')
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_transposed(self):
        a = np.arange(16).reshape(4, 4)
        out = np.empty_like(a).T
        a.clip(4, 10, out=out)
        expected = self.clip(a, 4, 10)
        assert_array_equal(out, expected)

    def test_clip_with_out_memory_overlap(self):
        a = np.arange(16).reshape(4, 4)
        ac = a.copy()
        a[:-1].clip(4, 10, out=a[1:])
        expected = self.clip(ac[:-1], 4, 10)
        assert_array_equal(a[1:], expected)

    def test_clip_inplace_array(self):
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_inplace_simple(self):
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_func_takes_out(self):
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        a2 = np.clip(a, m, M, out=a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a2, ac)
        assert_(a2 is a)

    def test_clip_nan(self):
        d = np.arange(7.0)
        assert_equal(d.clip(min=np.nan), np.nan)
        assert_equal(d.clip(max=np.nan), np.nan)
        assert_equal(d.clip(min=np.nan, max=np.nan), np.nan)
        assert_equal(d.clip(min=-2, max=np.nan), np.nan)
        assert_equal(d.clip(min=np.nan, max=10), np.nan)

    def test_object_clip(self):
        a = np.arange(10, dtype=object)
        actual = np.clip(a, 1, 5)
        expected = np.array([1, 1, 2, 3, 4, 5, 5, 5, 5, 5])
        assert actual.tolist() == expected.tolist()

    def test_clip_all_none(self):
        a = np.arange(10, dtype=object)
        with assert_raises_regex(ValueError, 'max or min'):
            np.clip(a, None, None)

    def test_clip_invalid_casting(self):
        a = np.arange(10, dtype=object)
        with assert_raises_regex(ValueError, 'casting must be one of'):
            self.fastclip(a, 1, 8, casting='garbage')

    @pytest.mark.parametrize('amin, amax', [(1, 0), (1, np.zeros(10)), (np.ones(10), np.zeros(10))])
    def test_clip_value_min_max_flip(self, amin, amax):
        a = np.arange(10, dtype=np.int64)
        expected = np.minimum(np.maximum(a, amin), amax)
        actual = np.clip(a, amin, amax)
        assert_equal(actual, expected)

    @pytest.mark.parametrize('arr, amin, amax, exp', [(np.zeros(10, dtype=np.int64), 0, -2 ** 64 + 1, np.full(10, -2 ** 64 + 1, dtype=object)), (np.zeros(10, dtype='m8') - 1, 0, 0, np.zeros(10, dtype='m8'))])
    def test_clip_problem_cases(self, arr, amin, amax, exp):
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, exp)

    @pytest.mark.parametrize('arr, amin, amax', [(np.zeros(10, dtype=np.int64), np.array(np.nan), np.zeros(10, dtype=np.int32))])
    def test_clip_scalar_nan_propagation(self, arr, amin, amax):
        expected = np.minimum(np.maximum(arr, amin), amax)
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, expected)

    @pytest.mark.xfail(reason="propagation doesn't match spec")
    @pytest.mark.parametrize('arr, amin, amax', [(np.array([1] * 10, dtype='m8'), np.timedelta64('NaT'), np.zeros(10, dtype=np.int32))])
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_NaT_propagation(self, arr, amin, amax):
        expected = np.minimum(np.maximum(arr, amin), amax)
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, expected)

    @given(data=st.data(), arr=hynp.arrays(dtype=hynp.integer_dtypes() | hynp.floating_dtypes(), shape=hynp.array_shapes()))
    def test_clip_property(self, data, arr):
        """A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        """
        numeric_dtypes = hynp.integer_dtypes() | hynp.floating_dtypes()
        in_shapes, result_shape = data.draw(hynp.mutually_broadcastable_shapes(num_shapes=2, base_shape=arr.shape))
        s = numeric_dtypes.flatmap(lambda x: hynp.from_dtype(x, allow_nan=False))
        amin = data.draw(s | hynp.arrays(dtype=numeric_dtypes, shape=in_shapes[0], elements={'allow_nan': False}))
        amax = data.draw(s | hynp.arrays(dtype=numeric_dtypes, shape=in_shapes[1], elements={'allow_nan': False}))
        result = np.clip(arr, amin, amax)
        t = np.result_type(arr, amin, amax)
        expected = np.minimum(amax, np.maximum(arr, amin, dtype=t), dtype=t)
        assert result.dtype == t
        assert_array_equal(result, expected)