import pytest
from numpy import (
from numpy.testing import (
def test_start_stop_array_scalar(self):
    lim1 = array([-120, 100], dtype='int8')
    lim2 = array([120, -100], dtype='int8')
    lim3 = array([1200, 1000], dtype='uint16')
    t1 = linspace(lim1[0], lim1[1], 5)
    t2 = linspace(lim2[0], lim2[1], 5)
    t3 = linspace(lim3[0], lim3[1], 5)
    t4 = linspace(-120.0, 100.0, 5)
    t5 = linspace(120.0, -100.0, 5)
    t6 = linspace(1200.0, 1000.0, 5)
    assert_equal(t1, t4)
    assert_equal(t2, t5)
    assert_equal(t3, t6)