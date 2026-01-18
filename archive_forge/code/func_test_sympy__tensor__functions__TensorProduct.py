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
def test_sympy__tensor__functions__TensorProduct():
    from sympy.tensor.functions import TensorProduct
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    tp = TensorProduct(A, B)
    assert _test_args(tp)