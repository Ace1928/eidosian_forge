import pytest
from numpy import (
from numpy.testing import (
def test_start_stop_array(self):
    start = array([-120, 120], dtype='int8')
    stop = array([100, -100], dtype='int8')
    t1 = linspace(start, stop, 5)
    t2 = stack([linspace(_start, _stop, 5) for _start, _stop in zip(start, stop)], axis=1)
    assert_equal(t1, t2)
    t3 = linspace(start, stop[0], 5)
    t4 = stack([linspace(_start, stop[0], 5) for _start in start], axis=1)
    assert_equal(t3, t4)
    t5 = linspace(start, stop, 5, axis=-1)
    assert_equal(t5, t2.T)