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
def _test_term(expr, x):
    """
        Test if ``expr`` fits the proper form for undetermined coefficients.
        """
    if not expr.has(x):
        return True
    elif expr.is_Add:
        return all((_test_term(i, x) for i in expr.args))
    elif expr.is_Mul:
        if expr.has(sin, cos):
            foundtrig = False
            for i in expr.args:
                if i.has(sin, cos):
                    if foundtrig:
                        return False
                    else:
                        foundtrig = True
        return all((_test_term(i, x) for i in expr.args))
    elif expr.is_Function:
        if expr.func in (sin, cos, exp, sinh, cosh):
            if expr.args[0].match(a * x + b):
                return True
            else:
                return False
        else:
            return False
    elif expr.is_Pow and expr.base.is_Symbol and expr.exp.is_Integer and (expr.exp >= 0):
        return True
    elif expr.is_Pow and expr.base.is_number:
        if expr.exp.match(a * x + b):
            return True
        else:
            return False
    elif expr.is_Symbol or expr.is_number:
        return True
    else:
        return False