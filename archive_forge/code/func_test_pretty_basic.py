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
def test_pretty_basic():
    assert pretty(-Rational(1) / 2) == '-1/2'
    assert pretty(-Rational(13) / 22) == '-13 \n----\n 22 '
    expr = oo
    ascii_str = 'oo'
    ucode_str = '∞'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = x ** 2
    ascii_str = ' 2\nx '
    ucode_str = ' 2\nx '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = 1 / x
    ascii_str = '1\n-\nx'
    ucode_str = '1\n─\nx'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = x ** (-1.0)
    ascii_str = ' -1.0\nx    '
    ucode_str = ' -1.0\nx    '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Pow(S(2), -1.0, evaluate=False)
    ascii_str = ' -1.0\n2    '
    ucode_str = ' -1.0\n2    '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = y * x ** (-2)
    ascii_str = 'y \n--\n 2\nx '
    ucode_str = 'y \n──\n 2\nx '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = x ** Rational(1, 3)
    ascii_str = ' 1/3\nx   '
    ucode_str = ' 1/3\nx   '
    assert xpretty(expr, use_unicode=False, wrap_line=False, root_notation=False) == ascii_str
    assert xpretty(expr, use_unicode=True, wrap_line=False, root_notation=False) == ucode_str
    expr = x ** Rational(-5, 2)
    ascii_str = ' 1  \n----\n 5/2\nx   '
    ucode_str = ' 1  \n────\n 5/2\nx   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = (-2) ** x
    ascii_str = '    x\n(-2) '
    ucode_str = '    x\n(-2) '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Pow(3, 1, evaluate=False)
    ascii_str = ' 1\n3 '
    ucode_str = ' 1\n3 '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = x ** 2 + x + 1
    ascii_str_1 = '         2\n1 + x + x '
    ascii_str_2 = ' 2        \nx  + x + 1'
    ascii_str_3 = ' 2        \nx  + 1 + x'
    ucode_str_1 = '         2\n1 + x + x '
    ucode_str_2 = ' 2        \nx  + x + 1'
    ucode_str_3 = ' 2        \nx  + 1 + x'
    assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]
    expr = 1 - x
    ascii_str_1 = '1 - x'
    ascii_str_2 = '-x + 1'
    ucode_str_1 = '1 - x'
    ucode_str_2 = '-x + 1'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = 1 - 2 * x
    ascii_str_1 = '1 - 2*x'
    ascii_str_2 = '-2*x + 1'
    ucode_str_1 = '1 - 2⋅x'
    ucode_str_2 = '-2⋅x + 1'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = x / y
    ascii_str = 'x\n-\ny'
    ucode_str = 'x\n─\ny'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -x / y
    ascii_str = '-x \n---\n y '
    ucode_str = '-x \n───\n y '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = (x + 2) / y
    ascii_str_1 = '2 + x\n-----\n  y  '
    ascii_str_2 = 'x + 2\n-----\n  y  '
    ucode_str_1 = '2 + x\n─────\n  y  '
    ucode_str_2 = 'x + 2\n─────\n  y  '
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = (1 + x) * y
    ascii_str_1 = 'y*(1 + x)'
    ascii_str_2 = '(1 + x)*y'
    ascii_str_3 = 'y*(x + 1)'
    ucode_str_1 = 'y⋅(1 + x)'
    ucode_str_2 = '(1 + x)⋅y'
    ucode_str_3 = 'y⋅(x + 1)'
    assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]
    expr = -5 * x / (x + 10)
    ascii_str_1 = '-5*x  \n------\n10 + x'
    ascii_str_2 = '-5*x  \n------\nx + 10'
    ucode_str_1 = '-5⋅x  \n──────\n10 + x'
    ucode_str_2 = '-5⋅x  \n──────\nx + 10'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = -S.Half - 3 * x
    ascii_str = '-3*x - 1/2'
    ucode_str = '-3⋅x - 1/2'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = S.Half - 3 * x
    ascii_str = '1/2 - 3*x'
    ucode_str = '1/2 - 3⋅x'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -S.Half - 3 * x / 2
    ascii_str = '  3*x   1\n- --- - -\n   2    2'
    ucode_str = '  3⋅x   1\n- ─── - ─\n   2    2'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = S.Half - 3 * x / 2
    ascii_str = '1   3*x\n- - ---\n2    2 '
    ucode_str = '1   3⋅x\n─ - ───\n2    2 '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str