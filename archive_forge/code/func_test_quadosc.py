import pytest
from mpmath import *
def test_quadosc():
    mp.dps = 15
    assert quadosc(lambda x: sin(x) / x, [0, inf], period=2 * pi).ae(pi / 2)