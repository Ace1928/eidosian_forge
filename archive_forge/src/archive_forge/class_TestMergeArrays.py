import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
class TestMergeArrays:

    def setup_method(self):
        x = np.array([1, 2])
        y = np.array([10, 20, 30])
        z = np.array([('A', 1.0), ('B', 2.0)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0, ())), (4, (5, 6.0, ()))], dtype=[('a', int), ('b', [('ba', float), ('bb', int), ('bc', [])])])
        self.data = (w, x, y, z)

    def test_solo(self):
        _, x, _, z = self.data
        test = merge_arrays(x)
        control = np.array([(1,), (2,)], dtype=[('f0', int)])
        assert_equal(test, control)
        test = merge_arrays((x,))
        assert_equal(test, control)
        test = merge_arrays(z, flatten=False)
        assert_equal(test, z)
        test = merge_arrays(z, flatten=True)
        assert_equal(test, z)

    def test_solo_w_flatten(self):
        w = self.data[0]
        test = merge_arrays(w, flatten=False)
        assert_equal(test, w)
        test = merge_arrays(w, flatten=True)
        control = np.array([(1, 2, 3.0), (4, 5, 6.0)], dtype=[('a', int), ('ba', float), ('bb', int)])
        assert_equal(test, control)

    def test_standard(self):
        _, x, y, _ = self.data
        test = merge_arrays((x, y), usemask=False)
        control = np.array([(1, 10), (2, 20), (-1, 30)], dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)
        test = merge_arrays((x, y), usemask=True)
        control = ma.array([(1, 10), (2, 20), (-1, 30)], mask=[(0, 0), (0, 0), (1, 0)], dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_flatten(self):
        _, x, _, z = self.data
        test = merge_arrays((x, z), flatten=True)
        control = np.array([(1, 'A', 1.0), (2, 'B', 2.0)], dtype=[('f0', int), ('A', '|S3'), ('B', float)])
        assert_equal(test, control)
        test = merge_arrays((x, z), flatten=False)
        control = np.array([(1, ('A', 1.0)), (2, ('B', 2.0))], dtype=[('f0', int), ('f1', [('A', '|S3'), ('B', float)])])
        assert_equal(test, control)

    def test_flatten_wflexible(self):
        w, x, _, _ = self.data
        test = merge_arrays((x, w), flatten=True)
        control = np.array([(1, 1, 2, 3.0), (2, 4, 5, 6.0)], dtype=[('f0', int), ('a', int), ('ba', float), ('bb', int)])
        assert_equal(test, control)
        test = merge_arrays((x, w), flatten=False)
        controldtype = [('f0', int), ('f1', [('a', int), ('b', [('ba', float), ('bb', int), ('bc', [])])])]
        control = np.array([(1.0, (1, (2, 3.0, ()))), (2, (4, (5, 6.0, ())))], dtype=controldtype)
        assert_equal(test, control)

    def test_wmasked_arrays(self):
        _, x, _, _ = self.data
        mx = ma.array([1, 2, 3], mask=[1, 0, 0])
        test = merge_arrays((x, mx), usemask=True)
        control = ma.array([(1, 1), (2, 2), (-1, 3)], mask=[(0, 1), (0, 0), (1, 0)], dtype=[('f0', int), ('f1', int)])
        assert_equal(test, control)
        test = merge_arrays((x, mx), usemask=True, asrecarray=True)
        assert_equal(test, control)
        assert_(isinstance(test, MaskedRecords))

    def test_w_singlefield(self):
        test = merge_arrays((np.array([1, 2]).view([('a', int)]), np.array([10.0, 20.0, 30.0])))
        control = ma.array([(1, 10.0), (2, 20.0), (-1, 30.0)], mask=[(0, 0), (0, 0), (1, 0)], dtype=[('a', int), ('f1', float)])
        assert_equal(test, control)

    def test_w_shorter_flex(self):
        z = self.data[-1]
        merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
        np.array([('A', 1.0, 10), ('B', 2.0, 20), ('-1', -1, 20)], dtype=[('A', '|S3'), ('B', float), ('C', int)])

    def test_singlerecord(self):
        _, x, y, z = self.data
        test = merge_arrays((x[0], y[0], z[0]), usemask=False)
        control = np.array([(1, 10, ('A', 1))], dtype=[('f0', int), ('f1', int), ('f2', [('A', '|S3'), ('B', float)])])
        assert_equal(test, control)