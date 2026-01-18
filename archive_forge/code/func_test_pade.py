import pytest
from mpmath import *
def test_pade():
    one = mpf(1)
    mp.dps = 20
    N = 10
    a = [one]
    k = 1
    for i in range(1, N + 1):
        k *= i
        a.append(one / k)
    p, q = pade(a, N // 2, N // 2)
    for x in arange(0, 1, 0.1):
        r = polyval(p[::-1], x) / polyval(q[::-1], x)
        assert r.ae(exp(x), 1e-10)
    mp.dps = 15