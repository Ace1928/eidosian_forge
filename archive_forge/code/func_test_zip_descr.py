import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_zip_descr(self):
    w, x, y, z = self.data
    test = zip_descr((x, x), flatten=True)
    assert_equal(test, np.dtype([('', int), ('', int)]))
    test = zip_descr((x, x), flatten=False)
    assert_equal(test, np.dtype([('', int), ('', int)]))
    test = zip_descr((x, z), flatten=True)
    assert_equal(test, np.dtype([('', int), ('A', '|S3'), ('B', float)]))
    test = zip_descr((x, z), flatten=False)
    assert_equal(test, np.dtype([('', int), ('', [('A', '|S3'), ('B', float)])]))
    test = zip_descr((x, w), flatten=True)
    assert_equal(test, np.dtype([('', int), ('a', int), ('ba', float), ('bb', int)]))
    test = zip_descr((x, w), flatten=False)
    assert_equal(test, np.dtype([('', int), ('', [('a', int), ('b', [('ba', float), ('bb', int)])])]))