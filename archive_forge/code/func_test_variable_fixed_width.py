import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_variable_fixed_width(self):
    strg = '  1     3  4  5  6# test'
    test = LineSplitter((3, 6, 6, 3))(strg)
    assert_equal(test, ['1', '3', '4  5', '6'])
    strg = '  1     3  4  5  6# test'
    test = LineSplitter((6, 6, 9))(strg)
    assert_equal(test, ['1', '3  4', '5  6'])