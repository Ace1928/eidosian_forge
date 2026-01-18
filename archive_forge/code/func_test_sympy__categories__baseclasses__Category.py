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
def test_sympy__categories__baseclasses__Category():
    from sympy.categories import Object, NamedMorphism, Diagram, Category
    A = Object('A')
    B = Object('B')
    C = Object('C')
    f = NamedMorphism(A, B, 'f')
    g = NamedMorphism(B, C, 'g')
    d1 = Diagram([f, g])
    d2 = Diagram([f])
    K = Category('K', commutative_diagrams=[d1, d2])
    assert _test_args(K)