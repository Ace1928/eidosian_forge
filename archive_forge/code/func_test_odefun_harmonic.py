from mpmath import odefun, cos, sin, mpf, sinc, mp
def test_odefun_harmonic():
    mp.dps = 15
    f = odefun(lambda x, y: [-y[1], y[0]], 0, [1, 0])
    for x in [0, 1, 2.5, 8, 3.7]:
        c, s = f(x)
        assert c.ae(cos(x))
        assert s.ae(sin(x))