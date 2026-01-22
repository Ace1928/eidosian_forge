import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
class TestMiscFunctions:

    def test_has_nested_dtype(self):
        """Test has_nested_dtype"""
        ndtype = np.dtype(float)
        assert_equal(has_nested_fields(ndtype), False)
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        assert_equal(has_nested_fields(ndtype), False)
        ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        assert_equal(has_nested_fields(ndtype), True)

    def test_easy_dtype(self):
        """Test ndtype on dtypes"""
        ndtype = float
        assert_equal(easy_dtype(ndtype), np.dtype(float))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype), np.dtype([('f0', 'i4'), ('f1', 'f8')]))
        assert_equal(easy_dtype(ndtype, defaultfmt='field_%03i'), np.dtype([('field_000', 'i4'), ('field_001', 'f8')]))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype, names='a, b'), np.dtype([('a', 'i4'), ('b', 'f8')]))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([('a', 'i4'), ('b', 'f8')]))
        ndtype = 'i4, f8'
        assert_equal(easy_dtype(ndtype, names=', b'), np.dtype([('f0', 'i4'), ('b', 'f8')]))
        assert_equal(easy_dtype(ndtype, names='a', defaultfmt='f%02i'), np.dtype([('a', 'i4'), ('f00', 'f8')]))
        ndtype = [('A', int), ('B', float)]
        assert_equal(easy_dtype(ndtype), np.dtype([('A', int), ('B', float)]))
        assert_equal(easy_dtype(ndtype, names='a,b'), np.dtype([('a', int), ('b', float)]))
        assert_equal(easy_dtype(ndtype, names='a'), np.dtype([('a', int), ('f0', float)]))
        assert_equal(easy_dtype(ndtype, names='a,b,c'), np.dtype([('a', int), ('b', float)]))
        ndtype = (int, float, float)
        assert_equal(easy_dtype(ndtype), np.dtype([('f0', int), ('f1', float), ('f2', float)]))
        ndtype = (int, float, float)
        assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([('a', int), ('b', float), ('c', float)]))
        ndtype = np.dtype(float)
        assert_equal(easy_dtype(ndtype, names='a, b, c'), np.dtype([(_, float) for _ in ('a', 'b', 'c')]))
        ndtype = np.dtype(float)
        assert_equal(easy_dtype(ndtype, names=['', '', ''], defaultfmt='f%02i'), np.dtype([(_, float) for _ in ('f00', 'f01', 'f02')]))

    def test_flatten_dtype(self):
        """Testing flatten_dtype"""
        dt = np.dtype([('a', 'f8'), ('b', 'f8')])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [float, float])
        dt = np.dtype([('a', [('aa', '|S1'), ('ab', '|S2')]), ('b', int)])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [np.dtype('|S1'), np.dtype('|S2'), int])
        dt = np.dtype([('a', (float, 2)), ('b', (int, 3))])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [float, int])
        dt_flat = flatten_dtype(dt, True)
        assert_equal(dt_flat, [float] * 2 + [int] * 3)
        dt = np.dtype([(('a', 'A'), 'f8'), (('b', 'B'), 'f8')])
        dt_flat = flatten_dtype(dt)
        assert_equal(dt_flat, [float, float])