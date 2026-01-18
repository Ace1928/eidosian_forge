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
@_both_exp_pow
def test_pretty_functions():
    """Tests for Abs, conjugate, exp, function braces, and factorial."""
    expr = 2 * x + exp(x)
    ascii_str_1 = '       x\n2*x + e '
    ascii_str_2 = ' x      \ne  + 2*x'
    ucode_str_1 = '       x\n2⋅x + ℯ '
    ucode_str_2 = ' x     \nℯ + 2⋅x'
    ucode_str_3 = ' x      \nℯ  + 2⋅x'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]
    expr = Abs(x)
    ascii_str = '|x|'
    ucode_str = '│x│'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Abs(x / (x ** 2 + 1))
    ascii_str_1 = '|  x   |\n|------|\n|     2|\n|1 + x |'
    ascii_str_2 = '|  x   |\n|------|\n| 2    |\n|x  + 1|'
    ucode_str_1 = '│  x   │\n│──────│\n│     2│\n│1 + x │'
    ucode_str_2 = '│  x   │\n│──────│\n│ 2    │\n│x  + 1│'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = Abs(1 / (y - Abs(x)))
    ascii_str = '    1    \n---------\n|y - |x||'
    ucode_str = '    1    \n─────────\n│y - │x││'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    n = Symbol('n', integer=True)
    expr = factorial(n)
    ascii_str = 'n!'
    ucode_str = 'n!'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = factorial(2 * n)
    ascii_str = '(2*n)!'
    ucode_str = '(2⋅n)!'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = factorial(factorial(factorial(n)))
    ascii_str = '((n!)!)!'
    ucode_str = '((n!)!)!'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = factorial(n + 1)
    ascii_str_1 = '(1 + n)!'
    ascii_str_2 = '(n + 1)!'
    ucode_str_1 = '(1 + n)!'
    ucode_str_2 = '(n + 1)!'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = subfactorial(n)
    ascii_str = '!n'
    ucode_str = '!n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = subfactorial(2 * n)
    ascii_str = '!(2*n)'
    ucode_str = '!(2⋅n)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    n = Symbol('n', integer=True)
    expr = factorial2(n)
    ascii_str = 'n!!'
    ucode_str = 'n!!'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = factorial2(2 * n)
    ascii_str = '(2*n)!!'
    ucode_str = '(2⋅n)!!'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = factorial2(factorial2(factorial2(n)))
    ascii_str = '((n!!)!!)!!'
    ucode_str = '((n!!)!!)!!'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = factorial2(n + 1)
    ascii_str_1 = '(1 + n)!!'
    ascii_str_2 = '(n + 1)!!'
    ucode_str_1 = '(1 + n)!!'
    ucode_str_2 = '(n + 1)!!'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = 2 * binomial(n, k)
    ascii_str = '  /n\\\n2*| |\n  \\k/'
    ucode_str = '  ⎛n⎞\n2⋅⎜ ⎟\n  ⎝k⎠'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = 2 * binomial(2 * n, k)
    ascii_str = '  /2*n\\\n2*|   |\n  \\ k /'
    ucode_str = '  ⎛2⋅n⎞\n2⋅⎜   ⎟\n  ⎝ k ⎠'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = 2 * binomial(n ** 2, k)
    ascii_str = '  / 2\\\n  |n |\n2*|  |\n  \\k /'
    ucode_str = '  ⎛ 2⎞\n  ⎜n ⎟\n2⋅⎜  ⎟\n  ⎝k ⎠'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = catalan(n)
    ascii_str = 'C \n n'
    ucode_str = 'C \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = catalan(n)
    ascii_str = 'C \n n'
    ucode_str = 'C \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = bell(n)
    ascii_str = 'B \n n'
    ucode_str = 'B \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = bernoulli(n)
    ascii_str = 'B \n n'
    ucode_str = 'B \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = bernoulli(n, x)
    ascii_str = 'B (x)\n n   '
    ucode_str = 'B (x)\n n   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = fibonacci(n)
    ascii_str = 'F \n n'
    ucode_str = 'F \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = lucas(n)
    ascii_str = 'L \n n'
    ucode_str = 'L \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = tribonacci(n)
    ascii_str = 'T \n n'
    ucode_str = 'T \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = stieltjes(n)
    ascii_str = 'stieltjes \n         n'
    ucode_str = 'γ \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = stieltjes(n, x)
    ascii_str = 'stieltjes (x)\n         n   '
    ucode_str = 'γ (x)\n n   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = mathieuc(x, y, z)
    ascii_str = 'C(x, y, z)'
    ucode_str = 'C(x, y, z)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = mathieus(x, y, z)
    ascii_str = 'S(x, y, z)'
    ucode_str = 'S(x, y, z)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = mathieucprime(x, y, z)
    ascii_str = "C'(x, y, z)"
    ucode_str = "C'(x, y, z)"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = mathieusprime(x, y, z)
    ascii_str = "S'(x, y, z)"
    ucode_str = "S'(x, y, z)"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = conjugate(x)
    ascii_str = '_\nx'
    ucode_str = '_\nx'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    f = Function('f')
    expr = conjugate(f(x + 1))
    ascii_str_1 = '________\nf(1 + x)'
    ascii_str_2 = '________\nf(x + 1)'
    ucode_str_1 = '________\nf(1 + x)'
    ucode_str_2 = '________\nf(x + 1)'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = f(x)
    ascii_str = 'f(x)'
    ucode_str = 'f(x)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = f(x, y)
    ascii_str = 'f(x, y)'
    ucode_str = 'f(x, y)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = f(x / (y + 1), y)
    ascii_str_1 = ' /  x     \\\nf|-----, y|\n \\1 + y   /'
    ascii_str_2 = ' /  x     \\\nf|-----, y|\n \\y + 1   /'
    ucode_str_1 = ' ⎛  x     ⎞\nf⎜─────, y⎟\n ⎝1 + y   ⎠'
    ucode_str_2 = ' ⎛  x     ⎞\nf⎜─────, y⎟\n ⎝y + 1   ⎠'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = f(x ** x ** x ** x ** x ** x)
    ascii_str = ' / / / / / x\\\\\\\\\\\n | | | | \\x /||||\n | | | \\x    /|||\n | | \\x       /||\n | \\x          /|\nf\\x             /'
    ucode_str = ' ⎛ ⎛ ⎛ ⎛ ⎛ x⎞⎞⎞⎞⎞\n ⎜ ⎜ ⎜ ⎜ ⎝x ⎠⎟⎟⎟⎟\n ⎜ ⎜ ⎜ ⎝x    ⎠⎟⎟⎟\n ⎜ ⎜ ⎝x       ⎠⎟⎟\n ⎜ ⎝x          ⎠⎟\nf⎝x             ⎠'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = sin(x) ** 2
    ascii_str = '   2   \nsin (x)'
    ucode_str = '   2   \nsin (x)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = conjugate(a + b * I)
    ascii_str = '_     _\na - I*b'
    ucode_str = '_     _\na - ⅈ⋅b'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = conjugate(exp(a + b * I))
    ascii_str = ' _     _\n a - I*b\ne       '
    ucode_str = ' _     _\n a - ⅈ⋅b\nℯ       '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = conjugate(f(1 + conjugate(f(x))))
    ascii_str_1 = '___________\n /    ____\\\nf\\1 + f(x)/'
    ascii_str_2 = '___________\n /____    \\\nf\\f(x) + 1/'
    ucode_str_1 = '___________\n ⎛    ____⎞\nf⎝1 + f(x)⎠'
    ucode_str_2 = '___________\n ⎛____    ⎞\nf⎝f(x) + 1⎠'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = f(x / (y + 1), y)
    ascii_str_1 = ' /  x     \\\nf|-----, y|\n \\1 + y   /'
    ascii_str_2 = ' /  x     \\\nf|-----, y|\n \\y + 1   /'
    ucode_str_1 = ' ⎛  x     ⎞\nf⎜─────, y⎟\n ⎝1 + y   ⎠'
    ucode_str_2 = ' ⎛  x     ⎞\nf⎜─────, y⎟\n ⎝y + 1   ⎠'
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]
    expr = floor(1 / (y - floor(x)))
    ascii_str = '     /     1      \\\nfloor|------------|\n     \\y - floor(x)/'
    ucode_str = '⎢   1   ⎥\n⎢───────⎥\n⎣y - ⌊x⌋⎦'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = ceiling(1 / (y - ceiling(x)))
    ascii_str = '       /      1       \\\nceiling|--------------|\n       \\y - ceiling(x)/'
    ucode_str = '⎡   1   ⎤\n⎢───────⎥\n⎢y - ⌈x⌉⎥'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = euler(n)
    ascii_str = 'E \n n'
    ucode_str = 'E \n n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = euler(1 / (1 + 1 / (1 + 1 / n)))
    ascii_str = 'E         \n     1    \n ---------\n       1  \n 1 + -----\n         1\n     1 + -\n         n'
    ucode_str = 'E         \n     1    \n ─────────\n       1  \n 1 + ─────\n         1\n     1 + ─\n         n'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = euler(n, x)
    ascii_str = 'E (x)\n n   '
    ucode_str = 'E (x)\n n   '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = euler(n, x / 2)
    ascii_str = '  /x\\\nE |-|\n n\\2/'
    ucode_str = '  ⎛x⎞\nE ⎜─⎟\n n⎝2⎠'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str