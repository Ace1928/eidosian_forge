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
def _solve_undetermined_coefficients(eq, func, order, match, trialset):
    """
    Helper function for the method of undetermined coefficients.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffUndeterminedCoefficients`
    docstring for more information on this method.

    The parameter ``trialset`` is the set of trial functions as returned by
    ``_undetermined_coefficients_match()['trialset']``.

    The parameter ``match`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
    r = match
    coeffs = numbered_symbols('a', cls=Dummy)
    coefflist = []
    gensols = r['list']
    gsol = r['sol']
    f = func.func
    x = func.args[0]
    if len(gensols) != order:
        raise NotImplementedError('Cannot find ' + str(order) + ' solutions to the homogeneous equation necessary to apply' + ' undetermined coefficients to ' + str(eq) + ' (number of terms != order)')
    trialfunc = 0
    for i in trialset:
        c = next(coeffs)
        coefflist.append(c)
        trialfunc += c * i
    eqs = sub_func_doit(eq, f(x), trialfunc)
    coeffsdict = dict(list(zip(trialset, [0] * (len(trialset) + 1))))
    eqs = _mexpand(eqs)
    for i in Add.make_args(eqs):
        s = separatevars(i, dict=True, symbols=[x])
        if coeffsdict.get(s[x]):
            coeffsdict[s[x]] += s['coeff']
        else:
            coeffsdict[s[x]] = s['coeff']
    coeffvals = solve(list(coeffsdict.values()), coefflist)
    if not coeffvals:
        raise NotImplementedError('Could not solve `%s` using the method of undetermined coefficients (unable to solve for coefficients).' % eq)
    psol = trialfunc.subs(coeffvals)
    return Eq(f(x), gsol.rhs + psol)