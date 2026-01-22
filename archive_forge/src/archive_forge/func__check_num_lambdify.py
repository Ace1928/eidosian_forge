import tempfile
from sympy.core.numbers import pi, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions import assuming, Q
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.cfunctions import log2, exp2, expm1, log1p
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.codegen.rewriting import (
from sympy.testing.pytest import XFAIL, skip
from sympy.utilities import lambdify
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
def _check_num_lambdify(expr, opt, val_subs, approx_ref, lambdify_kw=None, poorness=10000000000.0):
    """ poorness=1e10 signifies that `expr` loses precision of at least ten decimal digits. """
    num_ref = expr.subs(val_subs).evalf()
    eps = numpy.finfo(numpy.float64).eps
    assert abs(num_ref - approx_ref) < approx_ref * eps
    f1 = lambdify(list(val_subs.keys()), opt, **lambdify_kw or {})
    args_float = tuple(map(float, val_subs.values()))
    num_err1 = abs(f1(*args_float) - approx_ref)
    assert num_err1 < abs(num_ref * eps)
    f2 = lambdify(list(val_subs.keys()), expr, **lambdify_kw or {})
    num_err2 = abs(f2(*args_float) - approx_ref)
    assert num_err2 > abs(num_ref * eps * poorness)