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
def test_sympy__matrices__immutable__ImmutableSparseMatrix():
    from sympy.matrices.immutable import ImmutableSparseMatrix
    m = ImmutableSparseMatrix([[1, 2], [3, 4]])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(1, 1, {(0, 0): 1})
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(1, 1, [1])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(2, 2, lambda i, j: 1)
    assert m[0, 0] is S.One
    m = ImmutableSparseMatrix(2, 2, lambda i, j: 1 / (1 + i) + 1 / (1 + j))
    assert m[1, 1] is S.One
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))