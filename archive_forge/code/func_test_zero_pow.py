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
def test_zero_pow():
    assert satask(Q.zero(x ** y), Q.zero(x) & Q.positive(y)) is True
    assert satask(Q.zero(x ** y), Q.nonzero(x) & Q.zero(y)) is False
    assert satask(Q.zero(x), Q.zero(x ** y)) is True
    assert satask(Q.zero(x ** y), Q.zero(x)) is None