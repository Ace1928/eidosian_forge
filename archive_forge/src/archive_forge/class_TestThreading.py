import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
class TestThreading:

    def check_func_thread(self, n, fun, args, out):
        from threading import Thread
        thrds = [Thread(target=fun, args=args, kwargs={'output': out[x]}) for x in range(n)]
        [t.start() for t in thrds]
        [t.join() for t in thrds]

    def check_func_serial(self, n, fun, args, out):
        for i in range(n):
            fun(*args, output=out[i])

    def test_correlate1d(self):
        d = numpy.random.randn(5000)
        os = numpy.empty((4, d.size))
        ot = numpy.empty_like(os)
        k = numpy.arange(5)
        self.check_func_serial(4, ndimage.correlate1d, (d, k), os)
        self.check_func_thread(4, ndimage.correlate1d, (d, k), ot)
        assert_array_equal(os, ot)

    def test_correlate(self):
        d = numpy.random.randn(500, 500)
        k = numpy.random.randn(10, 10)
        os = numpy.empty([4] + list(d.shape))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.correlate, (d, k), os)
        self.check_func_thread(4, ndimage.correlate, (d, k), ot)
        assert_array_equal(os, ot)

    def test_median_filter(self):
        d = numpy.random.randn(500, 500)
        os = numpy.empty([4] + list(d.shape))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.median_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.median_filter, (d, 3), ot)
        assert_array_equal(os, ot)

    def test_uniform_filter1d(self):
        d = numpy.random.randn(5000)
        os = numpy.empty((4, d.size))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.uniform_filter1d, (d, 5), os)
        self.check_func_thread(4, ndimage.uniform_filter1d, (d, 5), ot)
        assert_array_equal(os, ot)

    def test_minmax_filter(self):
        d = numpy.random.randn(500, 500)
        os = numpy.empty([4] + list(d.shape))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.maximum_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.maximum_filter, (d, 3), ot)
        assert_array_equal(os, ot)
        self.check_func_serial(4, ndimage.minimum_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.minimum_filter, (d, 3), ot)
        assert_array_equal(os, ot)