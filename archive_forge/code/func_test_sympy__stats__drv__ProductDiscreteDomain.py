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
def test_sympy__stats__drv__ProductDiscreteDomain():
    from sympy.stats.drv import SingleDiscreteDomain, ProductDiscreteDomain
    X = SingleDiscreteDomain(x, S.Naturals)
    Y = SingleDiscreteDomain(y, S.Integers)
    assert _test_args(ProductDiscreteDomain(X, Y))