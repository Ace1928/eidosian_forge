from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.abc import alpha, beta, gamma, j, m
from sympy.physics.quantum import hbar, represent, Commutator, InnerProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import (
from sympy.testing.pytest import raises, slow
def test_uncouple_3_coupled_states():
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S(1) / 2, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S(1) / 2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S(1) / 2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S(1) / 2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S(1) / 2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S(1) / 2, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.NegativeOne / 2), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(1) / 2), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(1) / 2), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(1) / 2), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(1) / 2), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(1) / 2), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(1) / 2), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(-1) / 2), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(-1) / 2), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(-1) / 2), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(-1) / 2), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S(-1) / 2), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)))))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.NegativeOne / 2), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)))))