from sympy.assumptions.ask import Q
from sympy.assumptions.assume import assuming
from sympy.core.numbers import (I, pi)
from sympy.core.relational import (Eq, Gt)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import Implies
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.satask import (satask, extract_predargs,
from sympy.testing.pytest import raises, XFAIL
def test_rational_irrational():
    assert satask(Q.irrational(2)) is False
    assert satask(Q.rational(2)) is True
    assert satask(Q.irrational(pi)) is True
    assert satask(Q.rational(pi)) is False
    assert satask(Q.irrational(I)) is False
    assert satask(Q.rational(I)) is False
    assert satask(Q.irrational(x * y * z), Q.irrational(x) & Q.irrational(y) & Q.rational(z)) is None
    assert satask(Q.irrational(x * y * z), Q.irrational(x) & Q.rational(y) & Q.rational(z)) is True
    assert satask(Q.irrational(pi * x * y), Q.rational(x) & Q.rational(y)) is True
    assert satask(Q.irrational(x + y + z), Q.irrational(x) & Q.irrational(y) & Q.rational(z)) is None
    assert satask(Q.irrational(x + y + z), Q.irrational(x) & Q.rational(y) & Q.rational(z)) is True
    assert satask(Q.irrational(pi + x + y), Q.rational(x) & Q.rational(y)) is True
    assert satask(Q.irrational(x * y * z), Q.rational(x) & Q.rational(y) & Q.rational(z)) is False
    assert satask(Q.rational(x * y * z), Q.rational(x) & Q.rational(y) & Q.rational(z)) is True
    assert satask(Q.irrational(x + y + z), Q.rational(x) & Q.rational(y) & Q.rational(z)) is False
    assert satask(Q.rational(x + y + z), Q.rational(x) & Q.rational(y) & Q.rational(z)) is True