from mpmath import *
from mpmath.libmp import *
def test_round_down():
    assert from_man_exp(0, -4, 4, round_down)[:3] == (0, 0, 0)
    assert from_man_exp(240, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(241, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(255, -4, 4, round_down)[:3] == (0, 15, 0)
    assert from_man_exp(-240, -4, 4, round_down)[:3] == (1, 15, 0)
    assert from_man_exp(-241, -4, 4, round_down)[:3] == (1, 15, 0)
    assert from_man_exp(-255, -4, 4, round_down)[:3] == (1, 15, 0)