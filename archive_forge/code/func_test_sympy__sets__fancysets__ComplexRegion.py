import os
import re
from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import SKIP
from sympy.stats.crv_types import NormalDistribution
from sympy.stats.frv_types import DieDistribution
from sympy.matrices.expressions import MatrixSymbol
def test_sympy__sets__fancysets__ComplexRegion():
    from sympy.sets.fancysets import ComplexRegion
    from sympy.core.singleton import S
    from sympy.sets import Interval
    a = Interval(0, 1)
    b = Interval(2, 3)
    theta = Interval(0, 2 * S.Pi)
    assert _test_args(ComplexRegion(a * b))
    assert _test_args(ComplexRegion(a * theta, polar=True))