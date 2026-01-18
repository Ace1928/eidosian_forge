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
def test_represent_rotation():
    assert represent(Rotation(0, pi / 2, 0)) == Matrix([[WignerD(S(1) / 2, S(1) / 2, S(1) / 2, 0, pi / 2, 0), WignerD(S.Half, S.Half, Rational(-1, 2), 0, pi / 2, 0)], [WignerD(S.Half, Rational(-1, 2), S.Half, 0, pi / 2, 0), WignerD(S.Half, Rational(-1, 2), Rational(-1, 2), 0, pi / 2, 0)]])
    assert represent(Rotation(0, pi / 2, 0), doit=True) == Matrix([[sqrt(2) / 2, -sqrt(2) / 2], [sqrt(2) / 2, sqrt(2) / 2]])