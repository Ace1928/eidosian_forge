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
def test_zero_positive():
    assert satask(Q.zero(x + y), Q.positive(x) & Q.positive(y)) is False
    assert satask(Q.positive(x) & Q.positive(y), Q.zero(x + y)) is False
    assert satask(Q.nonzero(x + y), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.positive(x) & Q.positive(y), Q.nonzero(x + y)) is None
    assert satask(Q.zero(x * (x + y)), Q.positive(x) & Q.positive(y)) is False
    assert satask(Q.positive(pi * x * y + 1), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.positive(pi * x * y - 5), Q.positive(x) & Q.positive(y)) is None