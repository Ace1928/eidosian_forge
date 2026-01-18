from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_zeta_int_bug():
    assert mpf_zeta_int(0, 10) == from_float(-0.5)