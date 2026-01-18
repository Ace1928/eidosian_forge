from sympy.core.function import diff
from sympy.core.function import expand
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, sin, atan2, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.algebras.quaternion import Quaternion
from sympy.testing.pytest import raises
from itertools import permutations, product
def test_to_euler_numerical_singilarities():

    def test_one_case(angles, seq):
        q = Quaternion.from_euler(angles, seq)
        assert q.to_euler(seq) == angles
    test_one_case((pi / 2, 0, 0), 'zyz')
    test_one_case((pi / 2, 0, 0), 'ZYZ')
    test_one_case((pi / 2, pi, 0), 'zyz')
    test_one_case((pi / 2, pi, 0), 'ZYZ')
    test_one_case((pi / 2, pi / 2, 0), 'zyx')
    test_one_case((pi / 2, -pi / 2, 0), 'zyx')
    test_one_case((pi / 2, pi / 2, 0), 'ZYX')
    test_one_case((pi / 2, -pi / 2, 0), 'ZYX')