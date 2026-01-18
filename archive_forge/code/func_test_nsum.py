from mpmath import *
def test_nsum():
    mp.dps = 15
    assert nsum(lambda x: x ** 2, [1, 3]) == 14
    assert nsum(lambda k: 1 / factorial(k), [0, inf]).ae(e)
    assert nsum(lambda k: (-1) ** (k + 1) / k, [1, inf]).ae(log(2))
    assert nsum(lambda k: (-1) ** (k + 1) / k ** 2, [1, inf]).ae(pi ** 2 / 12)
    assert nsum(lambda k: (-1) ** k / log(k), [2, inf]).ae(0.9242998972229388)
    assert nsum(lambda k: 1 / k ** 2, [1, inf]).ae(pi ** 2 / 6)
    assert nsum(lambda k: 2 ** k / fac(k), [0, inf]).ae(exp(2))
    assert nsum(lambda k: 1 / k ** 2, [4, inf], method='e').ae(0.2838229557371153)
    assert abs(fp.nsum(lambda k: 1 / k ** 4, [1, fp.inf]) - 1.082323233711138) < 1e-05
    assert abs(fp.nsum(lambda k: 1 / k ** 4, [1, fp.inf], method='e') - 1.082323233711138) < 0.0001