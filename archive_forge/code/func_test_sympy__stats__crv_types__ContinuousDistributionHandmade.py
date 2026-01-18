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
def test_sympy__stats__crv_types__ContinuousDistributionHandmade():
    from sympy.stats.crv_types import ContinuousDistributionHandmade
    from sympy.core.function import Lambda
    from sympy.sets.sets import Interval
    from sympy.abc import x
    assert _test_args(ContinuousDistributionHandmade(Lambda(x, 2 * x), Interval(0, 1)))