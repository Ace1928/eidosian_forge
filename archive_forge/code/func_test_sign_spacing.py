import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_sign_spacing(self):
    a = np.arange(4.0)
    b = np.array([1234000000.0])
    c = np.array([1.0 + 1j, 1.123456789 + 1.123456789j], dtype='c16')
    assert_equal(repr(a), 'array([0., 1., 2., 3.])')
    assert_equal(repr(np.array(1.0)), 'array(1.)')
    assert_equal(repr(b), 'array([1.234e+09])')
    assert_equal(repr(np.array([0.0])), 'array([0.])')
    assert_equal(repr(c), 'array([1.        +1.j        , 1.12345679+1.12345679j])')
    assert_equal(repr(np.array([0.0, -0.0])), 'array([ 0., -0.])')
    np.set_printoptions(sign=' ')
    assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
    assert_equal(repr(np.array(1.0)), 'array( 1.)')
    assert_equal(repr(b), 'array([ 1.234e+09])')
    assert_equal(repr(c), 'array([ 1.        +1.j        ,  1.12345679+1.12345679j])')
    assert_equal(repr(np.array([0.0, -0.0])), 'array([ 0., -0.])')
    np.set_printoptions(sign='+')
    assert_equal(repr(a), 'array([+0., +1., +2., +3.])')
    assert_equal(repr(np.array(1.0)), 'array(+1.)')
    assert_equal(repr(b), 'array([+1.234e+09])')
    assert_equal(repr(c), 'array([+1.        +1.j        , +1.12345679+1.12345679j])')
    np.set_printoptions(legacy='1.13')
    assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
    assert_equal(repr(b), 'array([  1.23400000e+09])')
    assert_equal(repr(-b), 'array([ -1.23400000e+09])')
    assert_equal(repr(np.array(1.0)), 'array(1.0)')
    assert_equal(repr(np.array([0.0])), 'array([ 0.])')
    assert_equal(repr(c), 'array([ 1.00000000+1.j        ,  1.12345679+1.12345679j])')
    assert_equal(str(np.array([-1.0, 10])), '[ -1.  10.]')
    assert_raises(TypeError, np.set_printoptions, wrongarg=True)