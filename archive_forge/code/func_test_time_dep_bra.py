from sympy.core.add import Add
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.state import (
from sympy.physics.quantum.hilbert import HilbertSpace
def test_time_dep_bra():
    b = TimeDepBra(0, t)
    assert isinstance(b, TimeDepBra)
    assert isinstance(b, BraBase)
    assert isinstance(b, StateBase)
    assert isinstance(b, QExpr)
    assert b.label == (Integer(0),)
    assert b.args == (Integer(0), t)
    assert b.time == t
    assert b.dual_class() == TimeDepKet
    assert b.dual == TimeDepKet(0, t)
    k = TimeDepBra(x, 0.5)
    assert k.label == (x,)
    assert k.args == (x, sympify(0.5))
    assert TimeDepBra() == TimeDepBra('psi', 't')