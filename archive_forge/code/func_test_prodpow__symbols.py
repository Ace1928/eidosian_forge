from .._util import prodpow
from ..util.testing import requires
@requires('sympy')
def test_prodpow__symbols():
    import sympy
    a, b = sympy.symbols('a b')
    exprs = prodpow([a, b], [[0, 1], [1, 2]])
    assert exprs[0] == b
    assert exprs[1] == a * b ** 2