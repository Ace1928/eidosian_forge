from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_polygamma_identities():
    mp.dps = 15
    psi0 = lambda z: psi(0, z)
    psi1 = lambda z: psi(1, z)
    psi2 = lambda z: psi(2, z)
    assert psi0(0.5).ae(-euler - 2 * log(2))
    assert psi0(1).ae(-euler)
    assert psi1(0.5).ae(0.5 * pi ** 2)
    assert psi1(1).ae(pi ** 2 / 6)
    assert psi1(0.25).ae(pi ** 2 + 8 * catalan)
    assert psi2(1).ae(-2 * apery)
    mp.dps = 20
    u = -182 * apery + 4 * sqrt(3) * pi ** 3
    mp.dps = 15
    assert psi(2, 5 / 6.0).ae(u)
    assert psi(3, 0.5).ae(pi ** 4)