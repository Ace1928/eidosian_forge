import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_summarize_2d(self):
    A = np.arange(1002).reshape(2, 501)
    strA = '[[   0    1    2 ...  498  499  500]\n [ 501  502  503 ...  999 1000 1001]]'
    assert_equal(str(A), strA)
    reprA = 'array([[   0,    1,    2, ...,  498,  499,  500],\n       [ 501,  502,  503, ...,  999, 1000, 1001]])'
    assert_equal(repr(A), reprA)