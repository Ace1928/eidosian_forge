import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_tab_delimiter(self):
    """Test tab delimiter"""
    strg = ' 1\t 2\t 3\t 4\t 5  6'
    test = LineSplitter('\t')(strg)
    assert_equal(test, ['1', '2', '3', '4', '5  6'])
    strg = ' 1  2\t 3  4\t 5  6'
    test = LineSplitter('\t')(strg)
    assert_equal(test, ['1  2', '3  4', '5  6'])