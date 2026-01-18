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
Kalman Filter example from Matthew Rocklin's SciPy 2013 talk.

    Talk titled: "Matrix Expressions and BLAS/LAPACK; SciPy 2013 Presentation"

    Video: https://pyvideo.org/scipy-2013/matrix-expressions-and-blaslapack-scipy-2013-pr.html

    Notes
    =====

    Equations are:

    new_mu = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
           = MatAdd(mu, MatMul(Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data))))
    new_Sigma = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma
              = MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, Transpose(H)), Inverse(MatAdd(R, MatMul(H*Sigma*Transpose(H)))), H, Sigma))

    