from mpmath import odefun, cos, sin, mpf, sinc, mp
def test_odefun_sinc_large():
    mp.dps = 15
    f = sinc
    g = odefun(lambda x, y: [(cos(x) - y[0]) / x], 1, [f(1)], tol=0.01, degree=5)
    assert abs(f(100) - g(100)[0]) / f(100) < 0.01