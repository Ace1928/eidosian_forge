from collections import defaultdict
from sympy.core import Add, S
from sympy.core.function import diff, expand, _mexpand, expand_mul
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Wild
from sympy.functions import exp, cos, cosh, im, log, re, sin, sinh, \
from sympy.integrals import Integral
from sympy.polys import (Poly, RootOf, rootof, roots)
from sympy.simplify import collect, simplify, separatevars, powsimp, trigsimp # type: ignore
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.matrices import wronskian
from .subscheck import sub_func_doit
from sympy.solvers.ode.ode import get_numbered_constants
def _get_simplified_sol(sol, func, collectterms):
    """
    Helper function which collects the solution on
    collectterms. Ideally this should be handled by odesimp.It is used
    only when the simplify is set to True in dsolve.

    The parameter ``collectterms`` is a list of tuple (i, reroot, imroot) where `i` is
    the multiplicity of the root, reroot is real part and imroot being the imaginary part.

    """
    f = func.func
    x = func.args[0]
    collectterms.sort(key=default_sort_key)
    collectterms.reverse()
    assert len(sol) == 1 and sol[0].lhs == f(x)
    sol = sol[0].rhs
    sol = expand_mul(sol)
    for i, reroot, imroot in collectterms:
        sol = collect(sol, x ** i * exp(reroot * x) * sin(abs(imroot) * x))
        sol = collect(sol, x ** i * exp(reroot * x) * cos(imroot * x))
    for i, reroot, imroot in collectterms:
        sol = collect(sol, x ** i * exp(reroot * x))
    sol = powsimp(sol)
    return Eq(f(x), sol)