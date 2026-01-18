from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.physics.quantum.operator import (Operator, UnitaryOperator,
from sympy.physics.quantum.state import Ket, Bra, Wavefunction
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import JzKet, JzBra
from sympy.physics.quantum.trace import Tr
from sympy.matrices import eye
def test_outer_product():
    k = Ket('k')
    b = Bra('b')
    op = OuterProduct(k, b)
    assert isinstance(op, OuterProduct)
    assert isinstance(op, Operator)
    assert op.ket == k
    assert op.bra == b
    assert op.label == (k, b)
    assert op.is_commutative is False
    op = k * b
    assert isinstance(op, OuterProduct)
    assert isinstance(op, Operator)
    assert op.ket == k
    assert op.bra == b
    assert op.label == (k, b)
    assert op.is_commutative is False
    op = 2 * k * b
    assert op == Mul(Integer(2), k, b)
    op = 2 * (k * b)
    assert op == Mul(Integer(2), OuterProduct(k, b))
    assert Dagger(k * b) == OuterProduct(Dagger(b), Dagger(k))
    assert Dagger(k * b).is_commutative is False
    assert Tr(OuterProduct(JzKet(1, 1), JzBra(1, 1))).doit() == 1
    assert OuterProduct(2 * k, b) == 2 * OuterProduct(k, b)
    assert OuterProduct(k, 2 * b) == 2 * OuterProduct(k, b)
    k1, k2 = (Ket('k1'), Ket('k2'))
    b1, b2 = (Bra('b1'), Bra('b2'))
    assert OuterProduct(k1 + k2, b1) == OuterProduct(k1, b1) + OuterProduct(k2, b1)
    assert OuterProduct(k1, b1 + b2) == OuterProduct(k1, b1) + OuterProduct(k1, b2)
    assert OuterProduct(1 * k1 + 2 * k2, 3 * b1 + 4 * b2) == 3 * OuterProduct(k1, b1) + 4 * OuterProduct(k1, b2) + 6 * OuterProduct(k2, b1) + 8 * OuterProduct(k2, b2)