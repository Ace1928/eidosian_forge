import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
class TestIntFormat:

    def test_to_fortran(self):
        f = [IntFormat(10), IntFormat(12, 10), IntFormat(12, 10, 3)]
        res = ['(I10)', '(I12.10)', '(3I12.10)']
        for i, j in zip(f, res):
            assert_equal(i.fortran_format, j)

    def test_from_number(self):
        f = [10, -12, 123456789]
        r_f = [IntFormat(3, repeat=26), IntFormat(4, repeat=20), IntFormat(10, repeat=8)]
        for i, j in zip(f, r_f):
            assert_equal(IntFormat.from_number(i).__dict__, j.__dict__)