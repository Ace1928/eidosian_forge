from functools import reduce
import itertools
from operator import add
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import Inverse, MatAdd, MatMul, Transpose
from sympy.polys.rootoftools import CRootOf
from sympy.series.order import O
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import signsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.function import count_ops
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.functions.special.hyper import meijerg
from sympy.simplify import cse_main, cse_opts
from sympy.utilities.iterables import subsets
from sympy.testing.pytest import XFAIL, raises
from sympy.matrices import (MutableDenseMatrix, MutableSparseMatrix,
from sympy.matrices.expressions import MatrixSymbol
def test_subtraction_opt():
    e = (x - y) * (z - y) + exp((x - y) * (z - y))
    substs, reduced = cse([e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y) * (y - z))]
    assert reduced == [-x0 + exp(-x0)]
    e = -(x - y) * (z - y) + exp(-(x - y) * (z - y))
    substs, reduced = cse([e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y) * (y - z))]
    assert reduced == [x0 + exp(x0)]
    n = -1 + 1 / x
    e = n / x / (-n) ** 2 - 1 / n / x
    assert cse(e, optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)]) == ([], [0])
    assert cse((w + x + y + z) * (w - y - z) / (w + x) ** 3) == ([(x0, w + x), (x1, y + z)], [(w - x1) * (x0 + x1) / x0 ** 3])