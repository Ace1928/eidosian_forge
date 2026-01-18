import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_unnamed_and_named_fields(self):
    _, x, _, z = self.data
    test = stack_arrays((x, z))
    control = ma.array([(1, -1, -1), (2, -1, -1), (-1, 'A', 1), (-1, 'B', 2)], mask=[(0, 1, 1), (0, 1, 1), (1, 0, 0), (1, 0, 0)], dtype=[('f0', int), ('A', '|S3'), ('B', float)])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)
    test = stack_arrays((z, x))
    control = ma.array([('A', 1, -1), ('B', 2, -1), (-1, -1, 1), (-1, -1, 2)], mask=[(0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 0)], dtype=[('A', '|S3'), ('B', float), ('f2', int)])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)
    test = stack_arrays((z, z, x))
    control = ma.array([('A', 1, -1), ('B', 2, -1), ('A', 1, -1), ('B', 2, -1), (-1, -1, 1), (-1, -1, 2)], mask=[(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 0)], dtype=[('A', '|S3'), ('B', float), ('f2', int)])
    assert_equal(test, control)