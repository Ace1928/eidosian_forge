from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_siegelz():
    mp.dps = 15
    assert siegelz(100000).ae(5.879592468681765)
    assert siegelz(100000, derivative=2).ae(-54.11727110101265)
    assert siegelz(100000, derivative=3).ae(-278.93083134396653)
    assert siegelz(100000 + j, derivative=1).ae(678.2145118570703 - 379.7421607799164j)