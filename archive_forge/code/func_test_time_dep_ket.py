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
def test_time_dep_ket():
    k = TimeDepKet(0, t)
    assert isinstance(k, TimeDepKet)
    assert isinstance(k, KetBase)
    assert isinstance(k, StateBase)
    assert isinstance(k, QExpr)
    assert k.label == (Integer(0),)
    assert k.args == (Integer(0), t)
    assert k.time == t
    assert k.dual_class() == TimeDepBra
    assert k.dual == TimeDepBra(0, t)
    assert k.subs(t, 2) == TimeDepKet(0, 2)
    k = TimeDepKet(x, 0.5)
    assert k.label == (x,)
    assert k.args == (x, sympify(0.5))
    k = CustomTimeDepKet()
    assert k.label == (Symbol('test'),)
    assert k.time == Symbol('t')
    assert k == CustomTimeDepKet('test', 't')
    k = CustomTimeDepKetMultipleLabels()
    assert k.label == (Symbol('r'), Symbol('theta'), Symbol('phi'))
    assert k.time == Symbol('t')
    assert k == CustomTimeDepKetMultipleLabels('r', 'theta', 'phi', 't')
    assert TimeDepKet() == TimeDepKet('psi', 't')