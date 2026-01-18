from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core import (EulerGamma, GoldenRatio, Catalan)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.special.bessel import (airyai, airyaiprime, airybi, airybiprime)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import (fresnelc, fresnels)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import dirichlet_eta
from sympy.geometry.line import (Ray, Segment)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand, Nor, Not, Or, Xor)
from sympy.matrices.dense import (Matrix, diag)
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.trace import Trace
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR
from sympy.polys.orderings import (grlex, ilex)
from sympy.polys.polytools import groebner
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import O
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Complement, FiniteSet, Intersection, Interval, Union)
from sympy.codegen.ast import (Assignment, AddAugmentedAssignment,
from sympy.core.expr import UnevaluatedExpr
from sympy.physics.quantum.trace import Tr
from sympy.functions import (Abs, Chi, Ci, Ei, KroneckerDelta,
from sympy.matrices import (Adjoint, Inverse, MatrixSymbol, Transpose,
from sympy.matrices.expressions import hadamard_power
from sympy.physics import mechanics
from sympy.physics.control.lti import (TransferFunction, Feedback, TransferFunctionMatrix,
from sympy.physics.units import joule, degree
from sympy.printing.pretty import pprint, pretty as xpretty
from sympy.printing.pretty.pretty_symbology import center_accent, is_combining
from sympy.sets.conditionset import ConditionSet
from sympy.sets import ImageSet, ProductSet
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
from sympy.tensor.array import (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
from sympy.tensor.functions import TensorProduct
from sympy.tensor.tensor import (TensorIndexType, tensor_indices, TensorHead,
from sympy.testing.pytest import raises, _both_exp_pow, warns_deprecated_sympy
from sympy.vector import CoordSys3D, Gradient, Curl, Divergence, Dot, Cross, Laplacian
import sympy as sym
def test_pretty_ordering():
    assert pretty(x ** 2 + x + 1, order='lex') == ' 2        \nx  + x + 1'
    assert pretty(x ** 2 + x + 1, order='rev-lex') == '         2\n1 + x + x '
    assert pretty(1 - x, order='lex') == '-x + 1'
    assert pretty(1 - x, order='rev-lex') == '1 - x'
    assert pretty(1 - 2 * x, order='lex') == '-2*x + 1'
    assert pretty(1 - 2 * x, order='rev-lex') == '1 - 2*x'
    f = 2 * x ** 4 + y ** 2 - x ** 2 + y ** 3
    assert pretty(f, order=None) == '   4    2    3    2\n2*x  - x  + y  + y '
    assert pretty(f, order='lex') == '   4    2    3    2\n2*x  - x  + y  + y '
    assert pretty(f, order='rev-lex') == ' 2    3    2      4\ny  + y  - x  + 2*x '
    expr = x - x ** 3 / 6 + x ** 5 / 120 + O(x ** 6)
    ascii_str = '     3     5        \n    x     x     / 6\\\nx - -- + --- + O\\x /\n    6    120        '
    ucode_str = '     3     5        \n    x     x     ⎛ 6⎞\nx - ── + ─── + O⎝x ⎠\n    6    120        '
    assert pretty(expr, order=None) == ascii_str
    assert upretty(expr, order=None) == ucode_str
    assert pretty(expr, order='lex') == ascii_str
    assert upretty(expr, order='lex') == ucode_str
    assert pretty(expr, order='rev-lex') == ascii_str
    assert upretty(expr, order='rev-lex') == ucode_str