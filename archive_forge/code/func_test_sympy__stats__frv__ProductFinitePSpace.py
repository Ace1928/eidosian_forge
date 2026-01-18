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
def test_sympy__stats__frv__ProductFinitePSpace():
    from sympy.stats.frv import SingleFinitePSpace, ProductFinitePSpace
    from sympy.core.symbol import Symbol
    xp = SingleFinitePSpace(Symbol('x'), die)
    yp = SingleFinitePSpace(Symbol('y'), die)
    assert _test_args(ProductFinitePSpace(xp, yp))