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
class TestMaskedConstant:

    def _do_add_test(self, add):
        assert_(add(np.ma.masked, 1) is np.ma.masked)
        vector = np.array([1, 2, 3])
        result = add(np.ma.masked, vector)
        assert_(result is not np.ma.masked)
        assert_(not isinstance(result, np.ma.core.MaskedConstant))
        assert_equal(result.shape, vector.shape)
        assert_equal(np.ma.getmask(result), np.ones(vector.shape, dtype=bool))

    def test_ufunc(self):
        self._do_add_test(np.add)

    def test_operator(self):
        self._do_add_test(lambda a, b: a + b)

    def test_ctor(self):
        m = np.ma.array(np.ma.masked)
        assert_(not isinstance(m, np.ma.core.MaskedConstant))
        assert_(m is not np.ma.masked)

    def test_repr(self):
        assert_equal(repr(np.ma.masked), 'masked')
        masked2 = np.ma.MaskedArray.__new__(np.ma.core.MaskedConstant)
        assert_not_equal(repr(masked2), 'masked')

    def test_pickle(self):
        from io import BytesIO
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with BytesIO() as f:
                pickle.dump(np.ma.masked, f, protocol=proto)
                f.seek(0)
                res = pickle.load(f)
            assert_(res is np.ma.masked)

    def test_copy(self):
        assert_equal(np.ma.masked.copy() is np.ma.masked, np.True_.copy() is np.True_)

    def test__copy(self):
        import copy
        assert_(copy.copy(np.ma.masked) is np.ma.masked)

    def test_deepcopy(self):
        import copy
        assert_(copy.deepcopy(np.ma.masked) is np.ma.masked)

    def test_immutable(self):
        orig = np.ma.masked
        assert_raises(np.ma.core.MaskError, operator.setitem, orig, (), 1)
        assert_raises(ValueError, operator.setitem, orig.data, (), 1)
        assert_raises(ValueError, operator.setitem, orig.mask, (), False)
        view = np.ma.masked.view(np.ma.MaskedArray)
        assert_raises(ValueError, operator.setitem, view, (), 1)
        assert_raises(ValueError, operator.setitem, view.data, (), 1)
        assert_raises(ValueError, operator.setitem, view.mask, (), False)

    def test_coercion_int(self):
        a_i = np.zeros((), int)
        assert_raises(MaskError, operator.setitem, a_i, (), np.ma.masked)
        assert_raises(MaskError, int, np.ma.masked)

    def test_coercion_float(self):
        a_f = np.zeros((), float)
        assert_warns(UserWarning, operator.setitem, a_f, (), np.ma.masked)
        assert_(np.isnan(a_f[()]))

    @pytest.mark.xfail(reason='See gh-9750')
    def test_coercion_unicode(self):
        a_u = np.zeros((), 'U10')
        a_u[()] = np.ma.masked
        assert_equal(a_u[()], '--')

    @pytest.mark.xfail(reason='See gh-9750')
    def test_coercion_bytes(self):
        a_b = np.zeros((), 'S10')
        a_b[()] = np.ma.masked
        assert_equal(a_b[()], b'--')

    def test_subclass(self):

        class Sub(type(np.ma.masked)):
            pass
        a = Sub()
        assert_(a is Sub())
        assert_(a is not np.ma.masked)
        assert_not_equal(repr(a), 'masked')

    def test_attributes_readonly(self):
        assert_raises(AttributeError, setattr, np.ma.masked, 'shape', (1,))
        assert_raises(AttributeError, setattr, np.ma.masked, 'dtype', np.int64)