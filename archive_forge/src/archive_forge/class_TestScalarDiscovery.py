from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
class TestScalarDiscovery:

    def test_void_special_case(self):
        arr = np.array((1, 2, 3), dtype='i,i,i')
        assert arr.shape == ()
        arr = np.array([(1, 2, 3)], dtype='i,i,i')
        assert arr.shape == (1,)

    def test_char_special_case(self):
        arr = np.array('string', dtype='c')
        assert arr.shape == (6,)
        assert arr.dtype.char == 'c'
        arr = np.array(['string'], dtype='c')
        assert arr.shape == (1, 6)
        assert arr.dtype.char == 'c'

    def test_char_special_case_deep(self):
        nested = ['string']
        for i in range(np.MAXDIMS - 2):
            nested = [nested]
        arr = np.array(nested, dtype='c')
        assert arr.shape == (1,) * (np.MAXDIMS - 1) + (6,)
        with pytest.raises(ValueError):
            np.array([nested], dtype='c')

    def test_unknown_object(self):
        arr = np.array(object())
        assert arr.shape == ()
        assert arr.dtype == np.dtype('O')

    @pytest.mark.parametrize('scalar', scalar_instances())
    def test_scalar(self, scalar):
        arr = np.array(scalar)
        assert arr.shape == ()
        assert arr.dtype == scalar.dtype
        arr = np.array([[scalar, scalar]])
        assert arr.shape == (1, 2)
        assert arr.dtype == scalar.dtype

    @pytest.mark.filterwarnings('ignore:Promotion of numbers:FutureWarning')
    def test_scalar_promotion(self):
        for sc1, sc2 in product(scalar_instances(), scalar_instances()):
            sc1, sc2 = (sc1.values[0], sc2.values[0])
            try:
                arr = np.array([sc1, sc2])
            except (TypeError, ValueError):
                continue
            assert arr.shape == (2,)
            try:
                dt1, dt2 = (sc1.dtype, sc2.dtype)
                expected_dtype = np.promote_types(dt1, dt2)
                assert arr.dtype == expected_dtype
            except TypeError as e:
                assert arr.dtype == np.dtype('O')

    @pytest.mark.parametrize('scalar', scalar_instances())
    def test_scalar_coercion(self, scalar):
        if isinstance(scalar, np.inexact):
            scalar = type(scalar)((scalar * 2) ** 0.5)
        if type(scalar) is rational:
            pytest.xfail('Rational to object cast is undefined currently.')
        arr = np.array(scalar, dtype=object).astype(scalar.dtype)
        arr1 = np.array(scalar).reshape(1)
        arr2 = np.array([scalar])
        arr3 = np.empty(1, dtype=scalar.dtype)
        arr3[0] = scalar
        arr4 = np.empty(1, dtype=scalar.dtype)
        arr4[:] = [scalar]
        assert_array_equal(arr, arr1)
        assert_array_equal(arr, arr2)
        assert_array_equal(arr, arr3)
        assert_array_equal(arr, arr4)

    @pytest.mark.xfail(IS_PYPY, reason='`int(np.complex128(3))` fails on PyPy')
    @pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
    @pytest.mark.parametrize('cast_to', scalar_instances())
    def test_scalar_coercion_same_as_cast_and_assignment(self, cast_to):
        """
        Test that in most cases:
           * `np.array(scalar, dtype=dtype)`
           * `np.empty((), dtype=dtype)[()] = scalar`
           * `np.array(scalar).astype(dtype)`
        should behave the same.  The only exceptions are parametric dtypes
        (mainly datetime/timedelta without unit) and void without fields.
        """
        dtype = cast_to.dtype
        for scalar in scalar_instances(times=False):
            scalar = scalar.values[0]
            if dtype.type == np.void:
                if scalar.dtype.fields is not None and dtype.fields is None:
                    with pytest.raises(TypeError):
                        np.array(scalar).astype(dtype)
                    np.array(scalar, dtype=dtype)
                    np.array([scalar], dtype=dtype)
                    continue
            try:
                cast = np.array(scalar).astype(dtype)
            except (TypeError, ValueError, RuntimeError):
                with pytest.raises(Exception):
                    np.array(scalar, dtype=dtype)
                if isinstance(scalar, rational) and np.issubdtype(dtype, np.signedinteger):
                    return
                with pytest.raises(Exception):
                    np.array([scalar], dtype=dtype)
                res = np.zeros((), dtype=dtype)
                with pytest.raises(Exception):
                    res[()] = scalar
                return
            arr = np.array(scalar, dtype=dtype)
            assert_array_equal(arr, cast)
            ass = np.zeros((), dtype=dtype)
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize('pyscalar', [10, 10.32, 10.14j, 10 ** 100])
    def test_pyscalar_subclasses(self, pyscalar):
        """NumPy arrays are read/write which means that anything but invariant
        behaviour is on thin ice.  However, we currently are happy to discover
        subclasses of Python float, int, complex the same as the base classes.
        This should potentially be deprecated.
        """

        class MyScalar(type(pyscalar)):
            pass
        res = np.array(MyScalar(pyscalar))
        expected = np.array(pyscalar)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize('dtype_char', np.typecodes['All'])
    def test_default_dtype_instance(self, dtype_char):
        if dtype_char in 'SU':
            dtype = np.dtype(dtype_char + '1')
        elif dtype_char == 'V':
            dtype = np.dtype('V8')
        else:
            dtype = np.dtype(dtype_char)
        discovered_dtype, _ = _discover_array_parameters([], type(dtype))
        assert discovered_dtype == dtype
        assert discovered_dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize('dtype', np.typecodes['Integer'])
    @pytest.mark.parametrize(['scalar', 'error'], [(np.float64(np.nan), ValueError), (np.array(-1).astype(np.ulonglong)[()], OverflowError)])
    def test_scalar_to_int_coerce_does_not_cast(self, dtype, scalar, error):
        """
        Signed integers are currently different in that they do not cast other
        NumPy scalar, but instead use scalar.__int__(). The hardcoded
        exception to this rule is `np.array(scalar, dtype=integer)`.
        """
        dtype = np.dtype(dtype)
        with np.errstate(invalid='ignore'):
            coerced = np.array(scalar, dtype=dtype)
            cast = np.array(scalar).astype(dtype)
        assert_array_equal(coerced, cast)
        with pytest.raises(error):
            np.array([scalar], dtype=dtype)
        with pytest.raises(error):
            cast[()] = scalar