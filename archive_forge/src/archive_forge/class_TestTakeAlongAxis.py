import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestTakeAlongAxis:

    def test_argequivalent(self):
        """ Test it translates from arg<func> to <func> """
        from numpy.random import rand
        a = rand(3, 4, 5)
        funcs = [(np.sort, np.argsort, dict()), (_add_keepdims(np.min), _add_keepdims(np.argmin), dict()), (_add_keepdims(np.max), _add_keepdims(np.argmax), dict()), (np.partition, np.argpartition, dict(kth=2))]
        for func, argfunc, kwargs in funcs:
            for axis in list(range(a.ndim)) + [None]:
                a_func = func(a, axis=axis, **kwargs)
                ai_func = argfunc(a, axis=axis, **kwargs)
                assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))

    def test_invalid(self):
        """ Test it errors when indices has too few dimensions """
        a = np.ones((10, 10))
        ai = np.ones((10, 2), dtype=np.intp)
        take_along_axis(a, ai, axis=1)
        assert_raises(ValueError, take_along_axis, a, np.array(1), axis=1)
        assert_raises(IndexError, take_along_axis, a, ai.astype(bool), axis=1)
        assert_raises(IndexError, take_along_axis, a, ai.astype(float), axis=1)
        assert_raises(np.AxisError, take_along_axis, a, ai, axis=10)

    def test_empty(self):
        """ Test everything is ok with empty results, even with inserted dims """
        a = np.ones((3, 4, 5))
        ai = np.ones((3, 0, 5), dtype=np.intp)
        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, ai.shape)

    def test_broadcast(self):
        """ Test that non-indexing dimensions are broadcast in both directions """
        a = np.ones((3, 4, 1))
        ai = np.ones((1, 2, 5), dtype=np.intp)
        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, (3, 2, 5))