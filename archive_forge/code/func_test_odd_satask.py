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
def test_odd_satask():
    assert satask(Q.odd(2)) is False
    assert satask(Q.odd(3)) is True
    assert satask(Q.odd(x * y), Q.even(x) & Q.odd(y)) is False
    assert satask(Q.odd(x * y), Q.even(x) & Q.integer(y)) is False
    assert satask(Q.odd(x * y), Q.even(x) & Q.even(y)) is False
    assert satask(Q.odd(x * y), Q.odd(x) & Q.odd(y)) is True
    assert satask(Q.odd(x * y), Q.even(x)) is None
    assert satask(Q.odd(x * y), Q.odd(x) & Q.integer(y)) is None
    assert satask(Q.odd(x * y), Q.odd(x) & Q.odd(y)) is True
    assert satask(Q.odd(abs(x)), Q.even(x)) is False
    assert satask(Q.odd(abs(x)), Q.odd(x)) is True
    assert satask(Q.odd(x), Q.odd(abs(x))) is None