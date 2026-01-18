from itertools import product
import math
import inspect
import mpmath
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, cot, sin,
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely)
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.delta_functions import (Heaviside)
from sympy.functions.special.error_functions import (Ei, erf, erfc, fresnelc, fresnels, Si, Ci)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, polygamma)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, false, ITE, Not, Or, true)
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.tensor.array import derive_by_array, Array
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.lambdify import lambdify
from sympy.core.expr import UnevaluatedExpr
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, log10, hypot
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.functions.elementary.complexes import re, im, arg
from sympy.functions.special.polynomials import \
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.utilities.lambdify import implemented_function, lambdastr
from sympy.testing.pytest import skip
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.external import import_module
from sympy.functions.special.gamma_functions import uppergamma, lowergamma
import sympy
def test_scipy_polys():
    if not scipy:
        skip('scipy not installed')
    numpy.random.seed(0)
    params = symbols('n k a b')
    polys = [(chebyshevt, 1), (chebyshevu, 1), (legendre, 1), (hermite, 1), (laguerre, 1), (gegenbauer, 2), (assoc_legendre, 2), (assoc_laguerre, 2), (jacobi, 3)]
    msg = 'The random test of the function {func} with the arguments {args} had failed because the SymPy result {sympy_result} and SciPy result {scipy_result} had failed to converge within the tolerance {tol} (Actual absolute difference : {diff})'
    for sympy_fn, num_params in polys:
        args = params[:num_params] + (x,)
        f = lambdify(args, sympy_fn(*args))
        for _ in range(10):
            tn = numpy.random.randint(3, 10)
            tparams = tuple(numpy.random.uniform(0, 5, size=num_params - 1))
            tv = numpy.random.uniform(-10, 10) + 1j * numpy.random.uniform(-5, 5)
            if sympy_fn == hermite:
                tv = numpy.real(tv)
            if sympy_fn == assoc_legendre:
                tv = numpy.random.uniform(-1, 1)
                tparams = tuple(numpy.random.randint(1, tn, size=1))
            vals = (tn,) + tparams + (tv,)
            scipy_result = f(*vals)
            sympy_result = sympy_fn(*vals).evalf()
            atol = 1e-09 * (1 + abs(sympy_result))
            diff = abs(scipy_result - sympy_result)
            try:
                assert diff < atol
            except TypeError:
                raise AssertionError(msg.format(func=repr(sympy_fn), args=repr(vals), sympy_result=repr(sympy_result), scipy_result=repr(scipy_result), diff=diff, tol=atol))