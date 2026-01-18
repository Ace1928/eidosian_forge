from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_primezeta():
    mp.dps = 15
    assert primezeta(0.9).ae(1.8388316154446882 + 3.141592653589793j)
    assert primezeta(4).ae(0.07699313976424685)
    assert primezeta(1) == inf
    assert primezeta(inf) == 0
    assert isnan(primezeta(nan))