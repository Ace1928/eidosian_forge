import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_other_delimiter(self):
    """Test LineSplitter on delimiter"""
    strg = '1,2,3,4,,5'
    test = LineSplitter(',')(strg)
    assert_equal(test, ['1', '2', '3', '4', '', '5'])
    strg = ' 1,2,3,4,,5 # test'
    test = LineSplitter(',')(strg)
    assert_equal(test, ['1', '2', '3', '4', '', '5'])
    strg = b' 1,2,3,4,,5 % test'
    test = LineSplitter(delimiter=b',', comments=b'%')(strg)
    assert_equal(test, ['1', '2', '3', '4', '', '5'])