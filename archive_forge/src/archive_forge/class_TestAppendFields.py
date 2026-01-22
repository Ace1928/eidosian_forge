import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
class TestAppendFields:

    def setup_method(self):
        x = np.array([1, 2])
        y = np.array([10, 20, 30])
        z = np.array([('A', 1.0), ('B', 2.0)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_append_single(self):
        _, x, _, _ = self.data
        test = append_fields(x, 'A', data=[10, 20, 30])
        control = ma.array([(1, 10), (2, 20), (-1, 30)], mask=[(0, 0), (0, 0), (1, 0)], dtype=[('f0', int), ('A', int)])
        assert_equal(test, control)

    def test_append_double(self):
        _, x, _, _ = self.data
        test = append_fields(x, ('A', 'B'), data=[[10, 20, 30], [100, 200]])
        control = ma.array([(1, 10, 100), (2, 20, 200), (-1, 30, -1)], mask=[(0, 0, 0), (0, 0, 0), (1, 0, 1)], dtype=[('f0', int), ('A', int), ('B', int)])
        assert_equal(test, control)

    def test_append_on_flex(self):
        z = self.data[-1]
        test = append_fields(z, 'C', data=[10, 20, 30])
        control = ma.array([('A', 1.0, 10), ('B', 2.0, 20), (-1, -1.0, 30)], mask=[(0, 0, 0), (0, 0, 0), (1, 1, 0)], dtype=[('A', '|S3'), ('B', float), ('C', int)])
        assert_equal(test, control)

    def test_append_on_nested(self):
        w = self.data[0]
        test = append_fields(w, 'C', data=[10, 20, 30])
        control = ma.array([(1, (2, 3.0), 10), (4, (5, 6.0), 20), (-1, (-1, -1.0), 30)], mask=[(0, (0, 0), 0), (0, (0, 0), 0), (1, (1, 1), 0)], dtype=[('a', int), ('b', [('ba', float), ('bb', int)]), ('C', int)])
        assert_equal(test, control)