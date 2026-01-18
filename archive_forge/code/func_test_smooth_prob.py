from __future__ import division
import pytest
from preshed.counter import PreshCounter
import os
def test_smooth_prob():
    p = PreshCounter()
    for i in range(10):
        p.inc(100 - i, 1)
    for i in range(6):
        p.inc(90 - i, 2)
    for i in range(4):
        p.inc(80 - i, 3)
    for i in range(2):
        p.inc(70 - i, 5)
    for i in range(1):
        p.inc(60 - i, 8)
    assert p.total == 10 * 1 + 6 * 2 + 4 * 3 + 2 * 5 + 1 * 8
    assert p.prob(100) == 1.0 / p.total
    assert p.prob(200) == 0.0
    assert p.prob(60) == 8.0 / p.total
    p.smooth()
    assert p.smoother(1) < 1.0
    assert p.smoother(8) < 8.0
    assert p.prob(1000) < p.prob(100)
    for event, count in reversed(sorted(p, key=lambda it: it[1])):
        assert p.smoother(count) < count