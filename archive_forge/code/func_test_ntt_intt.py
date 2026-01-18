from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Symbol, symbols, I, Rational
from sympy.discrete import (fft, ifft, ntt, intt, fwht, ifwht,
from sympy.testing.pytest import raises
def test_ntt_intt():
    p = 7 * 17 * 2 ** 23 + 1
    q = 2 * 500000003 + 1
    r = 2 * 3 * 5 * 7
    assert all((tf(ls, p) == ls for tf in (ntt, intt) for ls in ([], [5])))
    ls = list(range(6))
    nls = [15, 801133602, 738493201, 334102277, 998244350, 849020224, 259751156, 12232587]
    assert ntt(ls, p) == nls
    assert intt(nls, p) == ls + [0] * 2
    ls = [1 + 2 * I, 3 + 4 * I, 5 + 6 * I]
    x = Symbol('x', integer=True)
    raises(TypeError, lambda: ntt(x, p))
    raises(ValueError, lambda: intt([x, 2 * x, 3 * x ** 2, 4 * x ** 3], p))
    raises(ValueError, lambda: intt(ls, p))
    raises(ValueError, lambda: ntt([1.2, 2.1, 3.5], p))
    raises(ValueError, lambda: ntt([3, 5, 6], q))
    raises(ValueError, lambda: ntt([4, 5, 7], r))
    raises(ValueError, lambda: ntt([1.0, 2.0, 3.0], p))