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
def test_sympy__sandbox__indexed_integrals__IndexedIntegral():
    from sympy.tensor import IndexedBase, Idx
    from sympy.sandbox.indexed_integrals import IndexedIntegral
    A = IndexedBase('A')
    i, j = symbols('i j', integer=True)
    a1, a2 = symbols('a1:3', cls=Idx)
    assert _test_args(IndexedIntegral(A[a1], A[a2]))
    assert _test_args(IndexedIntegral(A[i], A[j]))