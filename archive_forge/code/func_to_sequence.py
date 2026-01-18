from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve
from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
from sympy.integrals.meijerint import _mytype
def to_sequence(self, lb=True):
    """
        Finds recurrence relation for the coefficients in the series expansion
        of the function about :math:`x_0`, where :math:`x_0` is the point at
        which the initial condition is stored.

        Explanation
        ===========

        If the point :math:`x_0` is ordinary, solution of the form :math:`[(R, n_0)]`
        is returned. Where :math:`R` is the recurrence relation and :math:`n_0` is the
        smallest ``n`` for which the recurrence holds true.

        If the point :math:`x_0` is regular singular, a list of solutions in
        the format :math:`(R, p, n_0)` is returned, i.e. `[(R, p, n_0), ... ]`.
        Each tuple in this vector represents a recurrence relation :math:`R`
        associated with a root of the indicial equation ``p``. Conditions of
        a different format can also be provided in this case, see the
        docstring of HolonomicFunction class.

        If it's not possible to numerically compute a initial condition,
        it is returned as a symbol :math:`C_j`, denoting the coefficient of
        :math:`(x - x_0)^j` in the power series about :math:`x_0`.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
        [(HolonomicSequence((-1) + (n + 1)Sn, n), u(0) = 1, 0)]
        >>> HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]).to_sequence()
        [(HolonomicSequence((n**2) + (n**2 + n)Sn, n), u(0) = 0, u(1) = 1, u(2) = -1/2, 2)]
        >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_sequence()
        [(HolonomicSequence((n), n), u(0) = 1, 1/2, 1)]

        See Also
        ========

        HolonomicFunction.series

        References
        ==========

        .. [1] https://hal.inria.fr/inria-00070025/document
        .. [2] https://www3.risc.jku.at/publications/download/risc_2244/DIPLFORM.pdf

        """
    if self.x0 != 0:
        return self.shift_x(self.x0).to_sequence()
    if self.annihilator.is_singular(self.x0):
        return self._frobenius(lb=lb)
    dict1 = {}
    n = Symbol('n', integer=True)
    dom = self.annihilator.parent.base.dom
    R, _ = RecurrenceOperators(dom.old_poly_ring(n), 'Sn')
    for i, j in enumerate(self.annihilator.listofpoly):
        listofdmp = j.all_coeffs()
        degree = len(listofdmp) - 1
        for k in range(degree + 1):
            coeff = listofdmp[degree - k]
            if coeff == 0:
                continue
            if (i - k, k) in dict1:
                dict1[i - k, k] += dom.to_sympy(coeff) * rf(n - k + 1, i)
            else:
                dict1[i - k, k] = dom.to_sympy(coeff) * rf(n - k + 1, i)
    sol = []
    keylist = [i[0] for i in dict1]
    lower = min(keylist)
    upper = max(keylist)
    degree = self.degree()
    smallest_n = lower + degree
    dummys = {}
    eqs = []
    unknowns = []
    for j in range(lower, upper + 1):
        if j in keylist:
            temp = S.Zero
            for k in dict1.keys():
                if k[0] == j:
                    temp += dict1[k].subs(n, n - lower)
            sol.append(temp)
        else:
            sol.append(S.Zero)
    sol = RecurrenceOperator(sol, R)
    order = sol.order
    all_roots = roots(R.base.to_sympy(sol.listofpoly[-1]), n, filter='Z')
    all_roots = all_roots.keys()
    if all_roots:
        max_root = max(all_roots) + 1
        smallest_n = max(max_root, smallest_n)
    order += smallest_n
    y0 = _extend_y0(self, order)
    u0 = []
    for i, j in enumerate(y0):
        u0.append(j / factorial(i))
    if len(u0) < order:
        for i in range(degree):
            eq = S.Zero
            for j in dict1:
                if i + j[0] < 0:
                    dummys[i + j[0]] = S.Zero
                elif i + j[0] < len(u0):
                    dummys[i + j[0]] = u0[i + j[0]]
                elif not i + j[0] in dummys:
                    dummys[i + j[0]] = Symbol('C_%s' % (i + j[0]))
                    unknowns.append(dummys[i + j[0]])
                if j[1] <= i:
                    eq += dict1[j].subs(n, i) * dummys[i + j[0]]
            eqs.append(eq)
        soleqs = solve(eqs, *unknowns)
        if isinstance(soleqs, dict):
            for i in range(len(u0), order):
                if i not in dummys:
                    dummys[i] = Symbol('C_%s' % i)
                if dummys[i] in soleqs:
                    u0.append(soleqs[dummys[i]])
                else:
                    u0.append(dummys[i])
            if lb:
                return [(HolonomicSequence(sol, u0), smallest_n)]
            return [HolonomicSequence(sol, u0)]
        for i in range(len(u0), order):
            if i not in dummys:
                dummys[i] = Symbol('C_%s' % i)
            s = False
            for j in soleqs:
                if dummys[i] in j:
                    u0.append(j[dummys[i]])
                    s = True
            if not s:
                u0.append(dummys[i])
    if lb:
        return [(HolonomicSequence(sol, u0), smallest_n)]
    return [HolonomicSequence(sol, u0)]