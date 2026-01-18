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
def test_old_assump():
    assert satask(Q.positive(1)) is True
    assert satask(Q.positive(-1)) is False
    assert satask(Q.positive(0)) is False
    assert satask(Q.positive(I)) is False
    assert satask(Q.positive(pi)) is True
    assert satask(Q.negative(1)) is False
    assert satask(Q.negative(-1)) is True
    assert satask(Q.negative(0)) is False
    assert satask(Q.negative(I)) is False
    assert satask(Q.negative(pi)) is False
    assert satask(Q.zero(1)) is False
    assert satask(Q.zero(-1)) is False
    assert satask(Q.zero(0)) is True
    assert satask(Q.zero(I)) is False
    assert satask(Q.zero(pi)) is False
    assert satask(Q.nonzero(1)) is True
    assert satask(Q.nonzero(-1)) is True
    assert satask(Q.nonzero(0)) is False
    assert satask(Q.nonzero(I)) is False
    assert satask(Q.nonzero(pi)) is True
    assert satask(Q.nonpositive(1)) is False
    assert satask(Q.nonpositive(-1)) is True
    assert satask(Q.nonpositive(0)) is True
    assert satask(Q.nonpositive(I)) is False
    assert satask(Q.nonpositive(pi)) is False
    assert satask(Q.nonnegative(1)) is True
    assert satask(Q.nonnegative(-1)) is False
    assert satask(Q.nonnegative(0)) is True
    assert satask(Q.nonnegative(I)) is False
    assert satask(Q.nonnegative(pi)) is True