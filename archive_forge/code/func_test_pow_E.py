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
def test_pow_E():
    assert 2 ** (y / log(2)) == S.Exp1 ** y
    assert 2 ** (y / log(2) / 3) == S.Exp1 ** (y / 3)
    assert 3 ** (1 / log(-3)) != S.Exp1
    assert (3 + 2 * I) ** (1 / (log(-3 - 2 * I) + I * pi)) == S.Exp1
    assert (4 + 2 * I) ** (1 / (log(-4 - 2 * I) + I * pi)) == S.Exp1
    assert (3 + 2 * I) ** (1 / (log(-3 - 2 * I, 3) / 2 + I * pi / log(3) / 2)) == 9
    assert (3 + 2 * I) ** (1 / (log(3 + 2 * I, 3) / 2)) == 9
    while 1:
        b = x._random()
        r, i = b.as_real_imag()
        if i:
            break
    assert verify_numerically(b ** (1 / (log(-b) + sign(i) * I * pi).n()), S.Exp1)