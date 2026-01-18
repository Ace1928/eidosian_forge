import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_same_name_different_dtypes(self):
    a_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
    b_dtype = np.dtype([('key', 'S10'), ('value', '<f8')])
    expected_dtype = np.dtype([('key', '|S10'), ('value1', '<f4'), ('value2', '<f8')])
    a = np.array([('Sarah', 8.0), ('John', 6.0)], dtype=a_dtype)
    b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
    res = join_by('key', a, b)
    assert_equal(res.dtype, expected_dtype)