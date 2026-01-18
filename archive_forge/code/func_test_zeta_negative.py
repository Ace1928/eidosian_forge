from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_zeta_negative():
    mp.dps = 150
    a = -pi * 10 ** 40
    mp.dps = 15
    assert str(zeta(a)) == '2.55880492708712e+1233536161668617575553892558646631323374078'
    mp.dps = 50
    assert str(zeta(a)) == '2.5588049270871154960875033337384432038436330847333e+1233536161668617575553892558646631323374078'
    mp.dps = 15