from mpmath import *
from mpmath.libmp import *
def test_round_floor():
    assert from_man_exp(0, -4, 4, round_floor)[:3] == (0, 0, 0)
    assert from_man_exp(240, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(241, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(255, -4, 4, round_floor)[:3] == (0, 15, 0)
    assert from_man_exp(-240, -4, 4, round_floor)[:3] == (1, 15, 0)
    assert from_man_exp(-241, -4, 4, round_floor)[:3] == (1, 1, 4)
    assert from_man_exp(-255, -4, 4, round_floor)[:3] == (1, 1, 4)