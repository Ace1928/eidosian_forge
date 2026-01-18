from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet
from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin
from itertools import product
def test_pow_im():
    for m in (-2, -1, 2):
        for d in (3, 4, 5):
            b = m * I
            for i in range(1, 4 * d + 1):
                e = Rational(i, d)
                assert (b ** e - b.n() ** e.n()).n(2, chop=1e-10) == 0
    e = Rational(7, 3)
    assert (2 * x * I) ** e == 4 * 2 ** Rational(1, 3) * (I * x) ** e
    im = symbols('im', imaginary=True)
    assert (2 * im * I) ** e == 4 * 2 ** Rational(1, 3) * (I * im) ** e
    args = [I, I, I, I, 2]
    e = Rational(1, 3)
    ans = 2 ** e
    assert Mul(*args, evaluate=False) ** e == ans
    assert Mul(*args) ** e == ans
    args = [I, I, I, 2]
    e = Rational(1, 3)
    ans = 2 ** e * (-I) ** e
    assert Mul(*args, evaluate=False) ** e == ans
    assert Mul(*args) ** e == ans
    args.append(-3)
    ans = (6 * I) ** e
    assert Mul(*args, evaluate=False) ** e == ans
    assert Mul(*args) ** e == ans
    args.append(-1)
    ans = (-6 * I) ** e
    assert Mul(*args, evaluate=False) ** e == ans
    assert Mul(*args) ** e == ans
    args = [I, I, 2]
    e = Rational(1, 3)
    ans = (-2) ** e
    assert Mul(*args, evaluate=False) ** e == ans
    assert Mul(*args) ** e == ans
    args.append(-3)
    ans = 6 ** e
    assert Mul(*args, evaluate=False) ** e == ans
    assert Mul(*args) ** e == ans
    args.append(-1)
    ans = (-6) ** e
    assert Mul(*args, evaluate=False) ** e == ans
    assert Mul(*args) ** e == ans
    assert Mul(Pow(-1, Rational(3, 2), evaluate=False), I, I) == I
    assert Mul(I * Pow(I, S.Half, evaluate=False)) == sqrt(I) * I