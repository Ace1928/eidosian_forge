import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_ufunc_override_where(self):

    class OverriddenArrayOld(np.ndarray):

        def _unwrap(self, objs):
            cls = type(self)
            result = []
            for obj in objs:
                if isinstance(obj, cls):
                    obj = np.array(obj)
                elif type(obj) != np.ndarray:
                    return NotImplemented
                result.append(obj)
            return result

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            inputs = self._unwrap(inputs)
            if inputs is NotImplemented:
                return NotImplemented
            kwargs = kwargs.copy()
            if 'out' in kwargs:
                kwargs['out'] = self._unwrap(kwargs['out'])
                if kwargs['out'] is NotImplemented:
                    return NotImplemented
            r = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
            if r is not NotImplemented:
                r = r.view(type(self))
            return r

    class OverriddenArrayNew(OverriddenArrayOld):

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            kwargs = kwargs.copy()
            if 'where' in kwargs:
                kwargs['where'] = self._unwrap((kwargs['where'],))
                if kwargs['where'] is NotImplemented:
                    return NotImplemented
                else:
                    kwargs['where'] = kwargs['where'][0]
            r = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
            if r is not NotImplemented:
                r = r.view(type(self))
            return r
    ufunc = np.negative
    array = np.array([1, 2, 3])
    where = np.array([True, False, True])
    expected = ufunc(array, where=where)
    with pytest.raises(TypeError):
        ufunc(array, where=where.view(OverriddenArrayOld))
    result_1 = ufunc(array, where=where.view(OverriddenArrayNew))
    assert isinstance(result_1, OverriddenArrayNew)
    assert np.all(np.array(result_1) == expected, where=where)
    result_2 = ufunc(array.view(OverriddenArrayNew), where=where.view(OverriddenArrayNew))
    assert isinstance(result_2, OverriddenArrayNew)
    assert np.all(np.array(result_2) == expected, where=where)