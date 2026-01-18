from mpmath import *
from mpmath.libmp import *
def test_trig_near_zero():
    mp.dps = 15
    for r in [round_nearest, round_down, round_up, round_floor, round_ceiling]:
        assert sin(0, rounding=r) == 0
        assert cos(0, rounding=r) == 1
    a = mpf('1e-100')
    b = mpf('-1e-100')
    assert sin(a, rounding=round_nearest) == a
    assert sin(a, rounding=round_down) < a
    assert sin(a, rounding=round_floor) < a
    assert sin(a, rounding=round_up) >= a
    assert sin(a, rounding=round_ceiling) >= a
    assert sin(b, rounding=round_nearest) == b
    assert sin(b, rounding=round_down) > b
    assert sin(b, rounding=round_floor) <= b
    assert sin(b, rounding=round_up) <= b
    assert sin(b, rounding=round_ceiling) > b
    assert cos(a, rounding=round_nearest) == 1
    assert cos(a, rounding=round_down) < 1
    assert cos(a, rounding=round_floor) < 1
    assert cos(a, rounding=round_up) == 1
    assert cos(a, rounding=round_ceiling) == 1
    assert cos(b, rounding=round_nearest) == 1
    assert cos(b, rounding=round_down) < 1
    assert cos(b, rounding=round_floor) < 1
    assert cos(b, rounding=round_up) == 1
    assert cos(b, rounding=round_ceiling) == 1