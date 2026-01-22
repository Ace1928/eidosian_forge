import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestMaskedArray:

    def setup_method(self):
        x = np.array([1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0])
        a10 = 10.0
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        z = np.array([-0.5, 0.0, 0.5, 0.8])
        zm = masked_array(z, mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        xm.set_fill_value(1e+20)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)

    def test_basicattributes(self):
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a.ndim, 1)
        assert_equal(b.ndim, 1)
        assert_equal(a.size, 3)
        assert_equal(b.size, 3)
        assert_equal(a.shape, (3,))
        assert_equal(b.shape, (3,))

    def test_basic0d(self):
        x = masked_array(0)
        assert_equal(str(x), '0')
        x = masked_array(0, mask=True)
        assert_equal(str(x), str(masked_print_option))
        x = masked_array(0, mask=False)
        assert_equal(str(x), '0')
        x = array(0, mask=1)
        assert_(x.filled().dtype is x._data.dtype)

    def test_basic1d(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        assert_(not isMaskedArray(x))
        assert_(isMaskedArray(xm))
        assert_((xm - ym).filled(0).any())
        fail_if_equal(xm.mask.astype(int), ym.mask.astype(int))
        s = x.shape
        assert_equal(np.shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.dtype, x.dtype)
        assert_equal(zm.dtype, z.dtype)
        assert_equal(xm.size, reduce(lambda x, y: x * y, s))
        assert_equal(count(xm), len(m1) - reduce(lambda x, y: x + y, m1))
        assert_array_equal(xm, xf)
        assert_array_equal(filled(xm, 1e+20), xf)
        assert_array_equal(x, xm)

    def test_basic2d(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        for s in [(4, 3), (6, 2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s
            assert_(not isMaskedArray(x))
            assert_(isMaskedArray(xm))
            assert_equal(shape(xm), s)
            assert_equal(xm.shape, s)
            assert_equal(xm.size, reduce(lambda x, y: x * y, s))
            assert_equal(count(xm), len(m1) - reduce(lambda x, y: x + y, m1))
            assert_equal(xm, xf)
            assert_equal(filled(xm, 1e+20), xf)
            assert_equal(x, xm)

    def test_concatenate_basic(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        assert_equal(np.concatenate((x, y)), concatenate((xm, ym)))
        assert_equal(np.concatenate((x, y)), concatenate((x, y)))
        assert_equal(np.concatenate((x, y)), concatenate((xm, y)))
        assert_equal(np.concatenate((x, y, x)), concatenate((x, ym, x)))

    def test_concatenate_alongaxis(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        assert_equal(xm.mask, np.reshape(m1, s))
        assert_equal(ym.mask, np.reshape(m2, s))
        xmym = concatenate((xm, ym), 1)
        assert_equal(np.concatenate((x, y), 1), xmym)
        assert_equal(np.concatenate((xm.mask, ym.mask), 1), xmym._mask)
        x = zeros(2)
        y = array(ones(2), mask=[False, True])
        z = concatenate((x, y))
        assert_array_equal(z, [0, 0, 1, 1])
        assert_array_equal(z.mask, [False, False, False, True])
        z = concatenate((y, x))
        assert_array_equal(z, [1, 1, 0, 0])
        assert_array_equal(z.mask, [False, True, False, False])

    def test_concatenate_flexible(self):
        data = masked_array(list(zip(np.random.rand(10), np.arange(10))), dtype=[('a', float), ('b', int)])
        test = concatenate([data[:5], data[5:]])
        assert_equal_records(test, data)

    def test_creation_ndmin(self):
        x = array([1, 2, 3], mask=[1, 0, 0], ndmin=2)
        assert_equal(x.shape, (1, 3))
        assert_equal(x._data, [[1, 2, 3]])
        assert_equal(x._mask, [[1, 0, 0]])

    def test_creation_ndmin_from_maskedarray(self):
        x = array([1, 2, 3])
        x[-1] = masked
        xx = array(x, ndmin=2, dtype=float)
        assert_equal(x.shape, x._mask.shape)
        assert_equal(xx.shape, xx._mask.shape)

    def test_creation_maskcreation(self):
        data = arange(24, dtype=float)
        data[[3, 6, 15]] = masked
        dma_1 = MaskedArray(data)
        assert_equal(dma_1.mask, data.mask)
        dma_2 = MaskedArray(dma_1)
        assert_equal(dma_2.mask, dma_1.mask)
        dma_3 = MaskedArray(dma_1, mask=[1, 0, 0, 0] * 6)
        fail_if_equal(dma_3.mask, dma_1.mask)
        x = array([1, 2, 3], mask=True)
        assert_equal(x._mask, [True, True, True])
        x = array([1, 2, 3], mask=False)
        assert_equal(x._mask, [False, False, False])
        y = array([1, 2, 3], mask=x._mask, copy=False)
        assert_(np.may_share_memory(x.mask, y.mask))
        y = array([1, 2, 3], mask=x._mask, copy=True)
        assert_(not np.may_share_memory(x.mask, y.mask))
        x = array([1, 2, 3], mask=None)
        assert_equal(x._mask, [False, False, False])

    def test_masked_singleton_array_creation_warns(self):
        np.array(np.ma.masked)
        with pytest.warns(UserWarning):
            np.array([3.0, np.ma.masked])

    def test_creation_with_list_of_maskedarrays(self):
        x = array(np.arange(5), mask=[1, 0, 0, 0, 0])
        data = array((x, x[::-1]))
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        assert_equal(data._mask, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
        x.mask = nomask
        data = array((x, x[::-1]))
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        assert_(data.mask is nomask)

    def test_creation_with_list_of_maskedarrays_no_bool_cast(self):
        masked_str = np.ma.masked_array(['a', 'b'], mask=[True, False])
        normal_int = np.arange(2)
        res = np.ma.asarray([masked_str, normal_int], dtype='U21')
        assert_array_equal(res.mask, [[True, False], [False, False]])

        class NotBool:

            def __bool__(self):
                raise ValueError('not a bool!')
        masked_obj = np.ma.masked_array([NotBool(), 'b'], mask=[True, False])
        with pytest.raises(ValueError, match='not a bool!'):
            np.asarray([masked_obj], dtype=bool)
        res = np.ma.asarray([masked_obj, normal_int])
        assert_array_equal(res.mask, [[True, False], [False, False]])

    def test_creation_from_ndarray_with_padding(self):
        x = np.array([('A', 0)], dtype={'names': ['f0', 'f1'], 'formats': ['S4', 'i8'], 'offsets': [0, 8]})
        array(x)

    def test_unknown_keyword_parameter(self):
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            MaskedArray([1, 2, 3], maks=[0, 1, 0])

    def test_asarray(self):
        x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
        xm.fill_value = -9999
        xm._hardmask = True
        xmm = asarray(xm)
        assert_equal(xmm._data, xm._data)
        assert_equal(xmm._mask, xm._mask)
        assert_equal(xmm.fill_value, xm.fill_value)
        assert_equal(xmm._hardmask, xm._hardmask)

    def test_asarray_default_order(self):
        m = np.eye(3).T
        assert_(not m.flags.c_contiguous)
        new_m = asarray(m)
        assert_(new_m.flags.c_contiguous)

    def test_asarray_enforce_order(self):
        m = np.eye(3).T
        assert_(not m.flags.c_contiguous)
        new_m = asarray(m, order='C')
        assert_(new_m.flags.c_contiguous)

    def test_fix_invalid(self):
        with np.errstate(invalid='ignore'):
            data = masked_array([np.nan, 0.0, 1.0], mask=[0, 0, 1])
            data_fixed = fix_invalid(data)
            assert_equal(data_fixed._data, [data.fill_value, 0.0, 1.0])
            assert_equal(data_fixed._mask, [1.0, 0.0, 1.0])

    def test_maskedelement(self):
        x = arange(6)
        x[1] = masked
        assert_(str(masked) == '--')
        assert_(x[1] is masked)
        assert_equal(filled(x[1], 0), 0)

    def test_set_element_as_object(self):
        a = empty(1, dtype=object)
        x = (1, 2, 3, 4, 5)
        a[0] = x
        assert_equal(a[0], x)
        assert_(a[0] is x)
        import datetime
        dt = datetime.datetime.now()
        a[0] = dt
        assert_(a[0] is dt)

    def test_indexing(self):
        x1 = np.array([1, 2, 4, 3])
        x2 = array(x1, mask=[1, 0, 0, 0])
        x3 = array(x1, mask=[0, 1, 0, 1])
        x4 = array(x1)
        str(x2)
        repr(x2)
        assert_equal(np.sort(x1), sort(x2, endwith=False))
        assert_(type(x2[1]) is type(x1[1]))
        assert_(x1[1] == x2[1])
        assert_(x2[0] is masked)
        assert_equal(x1[2], x2[2])
        assert_equal(x1[2:5], x2[2:5])
        assert_equal(x1[:], x2[:])
        assert_equal(x1[1:], x3[1:])
        x1[2] = 9
        x2[2] = 9
        assert_equal(x1, x2)
        x1[1:3] = 99
        x2[1:3] = 99
        assert_equal(x1, x2)
        x2[1] = masked
        assert_equal(x1, x2)
        x2[1:3] = masked
        assert_equal(x1, x2)
        x2[:] = x1
        x2[1] = masked
        assert_(allequal(getmask(x2), array([0, 1, 0, 0])))
        x3[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x3), array([0, 1, 1, 0])))
        x4[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x4), array([0, 1, 1, 0])))
        assert_(allequal(x4, array([1, 2, 3, 4])))
        x1 = np.arange(5) * 1.0
        x2 = masked_values(x1, 3.0)
        assert_equal(x1, x2)
        assert_(allequal(array([0, 0, 0, 1, 0], MaskType), x2.mask))
        assert_equal(3.0, x2.fill_value)
        x1 = array([1, 'hello', 2, 3], object)
        x2 = np.array([1, 'hello', 2, 3], object)
        s1 = x1[1]
        s2 = x2[1]
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        assert_equal(s1, s2)
        assert_(x1[1:1].shape == (0,))

    def test_setitem_no_warning(self):
        x = np.ma.arange(60).reshape((6, 10))
        index = (slice(1, 5, 2), [7, 5])
        value = np.ma.masked_all((2, 2))
        value._data[...] = np.inf
        x[index] = value
        x[...] = np.ma.masked
        x = np.ma.arange(3.0, dtype=np.float32)
        value = np.ma.array([2e+234, 1, 1], mask=[True, False, False])
        x[...] = value
        x[[0, 1, 2]] = value

    @suppress_copy_mask_on_assignment
    def test_copy(self):
        n = [0, 0, 1, 0, 0]
        m = make_mask(n)
        m2 = make_mask(m)
        assert_(m is m2)
        m3 = make_mask(m, copy=True)
        assert_(m is not m3)
        x1 = np.arange(5)
        y1 = array(x1, mask=m)
        assert_equal(y1._data.__array_interface__, x1.__array_interface__)
        assert_(allequal(x1, y1.data))
        assert_equal(y1._mask.__array_interface__, m.__array_interface__)
        y1a = array(y1)
        assert_(y1a._data.__array_interface__ == y1._data.__array_interface__)
        assert_(y1a._mask.__array_interface__ == y1._mask.__array_interface__)
        y2 = array(x1, mask=m3)
        assert_(y2._data.__array_interface__ == x1.__array_interface__)
        assert_(y2._mask.__array_interface__ == m3.__array_interface__)
        assert_(y2[2] is masked)
        y2[2] = 9
        assert_(y2[2] is not masked)
        assert_(y2._mask.__array_interface__ == m3.__array_interface__)
        assert_(allequal(y2.mask, 0))
        y2a = array(x1, mask=m, copy=1)
        assert_(y2a._data.__array_interface__ != x1.__array_interface__)
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        assert_(y2a[2] is masked)
        y2a[2] = 9
        assert_(y2a[2] is not masked)
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        assert_(allequal(y2a.mask, 0))
        y3 = array(x1 * 1.0, mask=m)
        assert_(filled(y3).dtype is (x1 * 1.0).dtype)
        x4 = arange(4)
        x4[2] = masked
        y4 = resize(x4, (8,))
        assert_equal(concatenate([x4, x4]), y4)
        assert_equal(getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0])
        y5 = repeat(x4, (2, 2, 2, 2), axis=0)
        assert_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
        y6 = repeat(x4, 2, axis=0)
        assert_equal(y5, y6)
        y7 = x4.repeat((2, 2, 2, 2), axis=0)
        assert_equal(y5, y7)
        y8 = x4.repeat(2, 0)
        assert_equal(y5, y8)
        y9 = x4.copy()
        assert_equal(y9._data, x4._data)
        assert_equal(y9._mask, x4._mask)
        x = masked_array([1, 2, 3], mask=[0, 1, 0])
        y = masked_array(x)
        assert_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_equal(y._mask.ctypes.data, x._mask.ctypes.data)
        y = masked_array(x, copy=True)
        assert_not_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_not_equal(y._mask.ctypes.data, x._mask.ctypes.data)

    def test_copy_0d(self):
        x = np.ma.array(43, mask=True)
        xc = x.copy()
        assert_equal(xc.mask, True)

    def test_copy_on_python_builtins(self):
        assert_(isMaskedArray(np.ma.copy([1, 2, 3])))
        assert_(isMaskedArray(np.ma.copy((1, 2, 3))))

    def test_copy_immutable(self):
        a = np.ma.array([1, 2, 3])
        b = np.ma.array([4, 5, 6])
        a_copy_method = a.copy
        b.copy
        assert_equal(a_copy_method(), [1, 2, 3])

    def test_deepcopy(self):
        from copy import deepcopy
        a = array([0, 1, 2], mask=[False, True, False])
        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        assert_not_equal(id(a._mask), id(copied._mask))
        copied[1] = 1
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])
        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        copied.mask[1] = False
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])

    def test_format(self):
        a = array([0, 1, 2], mask=[False, True, False])
        assert_equal(format(a), '[0 -- 2]')
        assert_equal(format(masked), '--')
        assert_equal(format(masked, ''), '--')
        with assert_warns(FutureWarning):
            with_format_string = format(masked, ' >5')
        assert_equal(with_format_string, '--')

    def test_str_repr(self):
        a = array([0, 1, 2], mask=[False, True, False])
        assert_equal(str(a), '[0 -- 2]')
        assert_equal(repr(a), textwrap.dedent('            masked_array(data=[0, --, 2],\n                         mask=[False,  True, False],\n                   fill_value=999999)'))
        a = np.ma.arange(2000)
        a[1:50] = np.ma.masked
        assert_equal(repr(a), textwrap.dedent('            masked_array(data=[0, --, --, ..., 1997, 1998, 1999],\n                         mask=[False,  True,  True, ..., False, False, False],\n                   fill_value=999999)'))
        a = np.ma.arange(20)
        assert_equal(repr(a), textwrap.dedent('            masked_array(data=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,\n                               14, 15, 16, 17, 18, 19],\n                         mask=False,\n                   fill_value=999999)'))
        a = array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        a[1, 1] = np.ma.masked
        assert_equal(repr(a), textwrap.dedent('            masked_array(\n              data=[[1, 2, 3],\n                    [4, --, 6]],\n              mask=[[False, False, False],\n                    [False,  True, False]],\n              fill_value=999999,\n              dtype=int8)'))
        assert_equal(repr(a[:1]), textwrap.dedent('            masked_array(data=[[1, 2, 3]],\n                         mask=[[False, False, False]],\n                   fill_value=999999,\n                        dtype=int8)'))
        assert_equal(repr(a.astype(int)), textwrap.dedent('            masked_array(\n              data=[[1, 2, 3],\n                    [4, --, 6]],\n              mask=[[False, False, False],\n                    [False,  True, False]],\n              fill_value=999999)'))

    def test_str_repr_legacy(self):
        oldopts = np.get_printoptions()
        np.set_printoptions(legacy='1.13')
        try:
            a = array([0, 1, 2], mask=[False, True, False])
            assert_equal(str(a), '[0 -- 2]')
            assert_equal(repr(a), 'masked_array(data = [0 -- 2],\n             mask = [False  True False],\n       fill_value = 999999)\n')
            a = np.ma.arange(2000)
            a[1:50] = np.ma.masked
            assert_equal(repr(a), 'masked_array(data = [0 -- -- ..., 1997 1998 1999],\n             mask = [False  True  True ..., False False False],\n       fill_value = 999999)\n')
        finally:
            np.set_printoptions(**oldopts)

    def test_0d_unicode(self):
        u = 'café'
        utype = type(u)
        arr_nomask = np.ma.array(u)
        arr_masked = np.ma.array(u, mask=True)
        assert_equal(utype(arr_nomask), u)
        assert_equal(utype(arr_masked), '--')

    def test_pickling(self):
        for dtype in (int, float, str, object):
            a = arange(10).astype(dtype)
            a.fill_value = 999
            masks = ([0, 0, 0, 1, 0, 1, 0, 1, 0, 1], True, False)
            for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
                for mask in masks:
                    a.mask = mask
                    a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
                    assert_equal(a_pickled._mask, a._mask)
                    assert_equal(a_pickled._data, a._data)
                    if dtype in (object, int):
                        assert_equal(a_pickled.fill_value, 999)
                    else:
                        assert_equal(a_pickled.fill_value, dtype(999))
                    assert_array_equal(a_pickled.mask, mask)

    def test_pickling_subbaseclass(self):
        x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)]).view(np.recarray)
        a = masked_array(x, mask=[(True, False), (False, True)])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            assert_equal(a_pickled._mask, a._mask)
            assert_equal(a_pickled, a)
            assert_(isinstance(a_pickled._data, np.recarray))

    def test_pickling_maskedconstant(self):
        mc = np.ma.masked
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            mc_pickled = pickle.loads(pickle.dumps(mc, protocol=proto))
            assert_equal(mc_pickled._baseclass, mc._baseclass)
            assert_equal(mc_pickled._mask, mc._mask)
            assert_equal(mc_pickled._data, mc._data)

    def test_pickling_wstructured(self):
        a = array([(1, 1.0), (2, 2.0)], mask=[(0, 0), (0, 1)], dtype=[('a', int), ('b', float)])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            assert_equal(a_pickled._mask, a._mask)
            assert_equal(a_pickled, a)

    def test_pickling_keepalignment(self):
        a = arange(10)
        a.shape = (-1, 2)
        b = a.T
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            test = pickle.loads(pickle.dumps(b, protocol=proto))
            assert_equal(test, b)

    def test_single_element_subscript(self):
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a[0].shape, ())
        assert_equal(b[0].shape, ())
        assert_equal(b[1].shape, ())

    def test_topython(self):
        assert_equal(1, int(array(1)))
        assert_equal(1.0, float(array(1)))
        assert_equal(1, int(array([[[1]]])))
        assert_equal(1.0, float(array([[1]])))
        assert_raises(TypeError, float, array([1, 1]))
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Warning: converting a masked element')
            assert_(np.isnan(float(array([1], mask=[1]))))
            a = array([1, 2, 3], mask=[1, 0, 0])
            assert_raises(TypeError, lambda: float(a))
            assert_equal(float(a[-1]), 3.0)
            assert_(np.isnan(float(a[0])))
        assert_raises(TypeError, int, a)
        assert_equal(int(a[-1]), 3)
        assert_raises(MAError, lambda: int(a[0]))

    def test_oddfeatures_1(self):
        x = arange(20)
        x = x.reshape(4, 5)
        x.flat[5] = 12
        assert_(x[1, 0] == 12)
        z = x + 10j * x
        assert_equal(z.real, x)
        assert_equal(z.imag, 10 * x)
        assert_equal((z * conjugate(z)).real, 101 * x * x)
        z.imag[...] = 0.0
        x = arange(10)
        x[3] = masked
        assert_(str(x[3]) == str(masked))
        c = x >= 8
        assert_(count(where(c, masked, masked)) == 0)
        assert_(shape(where(c, masked, masked)) == c.shape)
        z = masked_where(c, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is not masked)
        assert_(z[7] is not masked)
        assert_(z[8] is masked)
        assert_(z[9] is masked)
        assert_equal(x, z)

    def test_oddfeatures_2(self):
        x = array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = array([1, 1, 1, 0, 0])
        x[2] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.0, 2.0, 0.0, -4.0, -5])
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)

    @suppress_copy_mask_on_assignment
    def test_oddfeatures_3(self):
        atest = array([10], mask=True)
        btest = array([20])
        idx = atest.mask
        atest[idx] = btest[idx]
        assert_equal(atest, [20])

    def test_filled_with_object_dtype(self):
        a = np.ma.masked_all(1, dtype='O')
        assert_equal(a.filled('x')[0], 'x')

    def test_filled_with_flexible_dtype(self):
        flexi = array([(1, 1, 1)], dtype=[('i', int), ('s', '|S8'), ('f', float)])
        flexi[0] = masked
        assert_equal(flexi.filled(), np.array([(default_fill_value(0), default_fill_value('0'), default_fill_value(0.0))], dtype=flexi.dtype))
        flexi[0] = masked
        assert_equal(flexi.filled(1), np.array([(1, '1', 1.0)], dtype=flexi.dtype))

    def test_filled_with_mvoid(self):
        ndtype = [('a', int), ('b', float)]
        a = mvoid((1, 2.0), mask=[(0, 1)], dtype=ndtype)
        test = a.filled()
        assert_equal(tuple(test), (1, default_fill_value(1.0)))
        test = a.filled((-1, -1))
        assert_equal(tuple(test), (1, -1))
        a.fill_value = (-999, -999)
        assert_equal(tuple(a.filled()), (1, -999))

    def test_filled_with_nested_dtype(self):
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([(1, (1, 1)), (2, (2, 2))], mask=[(0, (1, 0)), (0, (0, 1))], dtype=ndtype)
        test = a.filled(0)
        control = np.array([(1, (0, 1)), (2, (2, 0))], dtype=ndtype)
        assert_equal(test, control)
        test = a['B'].filled(0)
        control = np.array([(0, 1), (2, 0)], dtype=a['B'].dtype)
        assert_equal(test, control)
        Z = numpy.ma.zeros(2, numpy.dtype([('A', '(2,2)i1,(2,2)i1', (2, 2))]))
        assert_equal(Z.data.dtype, numpy.dtype([('A', [('f0', 'i1', (2, 2)), ('f1', 'i1', (2, 2))], (2, 2))]))
        assert_equal(Z.mask.dtype, numpy.dtype([('A', [('f0', '?', (2, 2)), ('f1', '?', (2, 2))], (2, 2))]))

    def test_filled_with_f_order(self):
        a = array(np.array([(0, 1, 2), (4, 5, 6)], order='F'), mask=np.array([(0, 0, 1), (1, 0, 0)], order='F'), order='F')
        assert_(a.flags['F_CONTIGUOUS'])
        assert_(a.filled(0).flags['F_CONTIGUOUS'])

    def test_optinfo_propagation(self):
        x = array([1, 2, 3], dtype=float)
        x._optinfo['info'] = '???'
        y = x.copy()
        assert_equal(y._optinfo['info'], '???')
        y._optinfo['info'] = '!!!'
        assert_equal(x._optinfo['info'], '???')

    def test_optinfo_forward_propagation(self):
        a = array([1, 2, 2, 4])
        a._optinfo['key'] = 'value'
        assert_equal(a._optinfo['key'], (a == 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a != 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a > 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a >= 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a <= 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a + 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a - 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a * 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], (a / 2)._optinfo['key'])
        assert_equal(a._optinfo['key'], a[:2]._optinfo['key'])
        assert_equal(a._optinfo['key'], a[[0, 0, 2]]._optinfo['key'])
        assert_equal(a._optinfo['key'], np.exp(a)._optinfo['key'])
        assert_equal(a._optinfo['key'], np.abs(a)._optinfo['key'])
        assert_equal(a._optinfo['key'], array(a, copy=True)._optinfo['key'])
        assert_equal(a._optinfo['key'], np.zeros_like(a)._optinfo['key'])

    def test_fancy_printoptions(self):
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = array([(1, (2, 3.0)), (4, (5, 6.0))], mask=[(1, (0, 1)), (0, (1, 0))], dtype=fancydtype)
        control = '[(--, (2, --)) (4, (--, 6.0))]'
        assert_equal(str(test), control)
        t_2d0 = masked_array(data=(0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 0.0), mask=(False, [[True, False, True], [False, False, True]], False), dtype='int, (2,3)float, float')
        control = '(0, [[--, 0.0, --], [0.0, 0.0, --]], 0.0)'
        assert_equal(str(t_2d0), control)

    def test_flatten_structured_array(self):
        ndtype = [('a', int), ('b', float)]
        a = np.array([(1, 1), (2, 2)], dtype=ndtype)
        test = flatten_structured_array(a)
        control = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
        test = flatten_structured_array(a)
        control = array([[1.0, 1.0], [2.0, 2.0]], mask=[[0, 1], [1, 0]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        assert_equal(test.mask, control.mask)
        ndtype = [('a', int), ('b', [('ba', int), ('bb', float)])]
        a = array([(1, (1, 1.1)), (2, (2, 2.2))], mask=[(0, (1, 0)), (1, (0, 1))], dtype=ndtype)
        test = flatten_structured_array(a)
        control = array([[1.0, 1.0, 1.1], [2.0, 2.0, 2.2]], mask=[[0, 1, 0], [1, 0, 1]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        assert_equal(test.mask, control.mask)
        ndtype = [('a', int), ('b', float)]
        a = np.array([[(1, 1)], [(2, 2)]], dtype=ndtype)
        test = flatten_structured_array(a)
        control = np.array([[[1.0, 1.0]], [[2.0, 2.0]]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)

    def test_void0d(self):
        ndtype = [('a', int), ('b', int)]
        a = np.array([(1, 2)], dtype=ndtype)[0]
        f = mvoid(a)
        assert_(isinstance(f, mvoid))
        a = masked_array([(1, 2)], mask=[(1, 0)], dtype=ndtype)[0]
        assert_(isinstance(a, mvoid))
        a = masked_array([(1, 2), (1, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
        f = mvoid(a._data[0], a._mask[0])
        assert_(isinstance(f, mvoid))

    def test_mvoid_getitem(self):
        ndtype = [('a', int), ('b', int)]
        a = masked_array([(1, 2), (3, 4)], mask=[(0, 0), (1, 0)], dtype=ndtype)
        f = a[0]
        assert_(isinstance(f, mvoid))
        assert_equal((f[0], f['a']), (1, 1))
        assert_equal(f['b'], 2)
        f = a[1]
        assert_(isinstance(f, mvoid))
        assert_(f[0] is masked)
        assert_(f['a'] is masked)
        assert_equal(f[1], 4)
        A = masked_array(data=[([0, 1],)], mask=[([True, False],)], dtype=[('A', '>i2', (2,))])
        assert_equal(A[0]['A'], A['A'][0])
        assert_equal(A[0]['A'], masked_array(data=[0, 1], mask=[True, False], dtype='>i2'))

    def test_mvoid_iter(self):
        ndtype = [('a', int), ('b', int)]
        a = masked_array([(1, 2), (3, 4)], mask=[(0, 0), (1, 0)], dtype=ndtype)
        assert_equal(list(a[0]), [1, 2])
        assert_equal(list(a[1]), [masked, 4])

    def test_mvoid_print(self):
        mx = array([(1, 1), (2, 2)], dtype=[('a', int), ('b', int)])
        assert_equal(str(mx[0]), '(1, 1)')
        mx['b'][0] = masked
        ini_display = masked_print_option._display
        masked_print_option.set_display('-X-')
        try:
            assert_equal(str(mx[0]), '(1, -X-)')
            assert_equal(repr(mx[0]), '(1, -X-)')
        finally:
            masked_print_option.set_display(ini_display)
        mx = array([(1,), (2,)], dtype=[('a', 'O')])
        assert_equal(str(mx[0]), '(1,)')

    def test_mvoid_multidim_print(self):
        t_ma = masked_array(data=[([1, 2, 3],)], mask=[([False, True, False],)], fill_value=([999999, 999999, 999999],), dtype=[('a', '<i4', (3,))])
        assert_(str(t_ma[0]) == '([1, --, 3],)')
        assert_(repr(t_ma[0]) == '([1, --, 3],)')
        t_2d = masked_array(data=[([[1, 2], [3, 4]],)], mask=[([[False, True], [True, False]],)], dtype=[('a', '<i4', (2, 2))])
        assert_(str(t_2d[0]) == '([[1, --], [--, 4]],)')
        assert_(repr(t_2d[0]) == '([[1, --], [--, 4]],)')
        t_0d = masked_array(data=[(1, 2)], mask=[(True, False)], dtype=[('a', '<i4'), ('b', '<i4')])
        assert_(str(t_0d[0]) == '(--, 2)')
        assert_(repr(t_0d[0]) == '(--, 2)')
        t_2d = masked_array(data=[([[1, 2], [3, 4]], 1)], mask=[([[False, True], [True, False]], False)], dtype=[('a', '<i4', (2, 2)), ('b', float)])
        assert_(str(t_2d[0]) == '([[1, --], [--, 4]], 1.0)')
        assert_(repr(t_2d[0]) == '([[1, --], [--, 4]], 1.0)')
        t_ne = masked_array(data=[(1, (1, 1))], mask=[(True, (True, False))], dtype=[('a', '<i4'), ('b', 'i4,i4')])
        assert_(str(t_ne[0]) == '(--, (--, 1))')
        assert_(repr(t_ne[0]) == '(--, (--, 1))')

    def test_object_with_array(self):
        mx1 = masked_array([1.0], mask=[True])
        mx2 = masked_array([1.0, 2.0])
        mx = masked_array([mx1, mx2], mask=[False, True], dtype=object)
        assert_(mx[0] is mx1)
        assert_(mx[1] is not mx2)
        assert_(np.all(mx[1].data == mx2.data))
        assert_(np.all(mx[1].mask))
        mx[1].data[0] = 0.0
        assert_(mx2[0] == 0.0)