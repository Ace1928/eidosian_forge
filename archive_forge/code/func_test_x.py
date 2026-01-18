from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval
from sympy.physics.quantum import qapply, represent, L2, Dagger
from sympy.physics.quantum import Commutator, hbar
from sympy.physics.quantum.cartesian import (
from sympy.physics.quantum.operator import DifferentialOperator
def test_x():
    assert X.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert Commutator(X, Px).doit() == I * hbar
    assert qapply(X * XKet(x)) == x * XKet(x)
    assert XKet(x).dual_class() == XBra
    assert XBra(x).dual_class() == XKet
    assert (Dagger(XKet(y)) * XKet(x)).doit() == DiracDelta(x - y)
    assert (PxBra(px) * XKet(x)).doit() == exp(-I * x * px / hbar) / sqrt(2 * pi * hbar)
    assert represent(XKet(x)) == DiracDelta(x - x_1)
    assert represent(XBra(x)) == DiracDelta(-x + x_1)
    assert XBra(x).position == x
    assert represent(XOp() * XKet()) == x * DiracDelta(x - x_2)
    assert represent(XOp() * XKet() * XBra('y')) == x * DiracDelta(x - x_3) * DiracDelta(x_1 - y)
    assert represent(XBra('y') * XKet()) == DiracDelta(x - y)
    assert represent(XKet() * XBra()) == DiracDelta(x - x_2) * DiracDelta(x_1 - x)
    rep_p = represent(XOp(), basis=PxOp)
    assert rep_p == hbar * I * DiracDelta(px_1 - px_2) * DifferentialOperator(px_1)
    assert rep_p == represent(XOp(), basis=PxOp())
    assert rep_p == represent(XOp(), basis=PxKet)
    assert rep_p == represent(XOp(), basis=PxKet())
    assert represent(XOp() * PxKet(), basis=PxKet) == hbar * I * DiracDelta(px - px_2) * DifferentialOperator(px)