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
def test_represent_spin_operators():
    assert represent(Jx) == hbar * Matrix([[0, 1], [1, 0]]) / 2
    assert represent(Jx, j=1) == hbar * sqrt(2) * Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 2
    assert represent(Jy) == hbar * I * Matrix([[0, -1], [1, 0]]) / 2
    assert represent(Jy, j=1) == hbar * I * sqrt(2) * Matrix([[0, -1, 0], [1, 0, -1], [0, 1, 0]]) / 2
    assert represent(Jz) == hbar * Matrix([[1, 0], [0, -1]]) / 2
    assert represent(Jz, j=1) == hbar * Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])