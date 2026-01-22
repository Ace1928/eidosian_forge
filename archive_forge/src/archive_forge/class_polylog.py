from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, Function
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly
class polylog(Function):
    """
    Polylogarithm function.

    Explanation
    ===========

    For $|z| < 1$ and $s \\in \\mathbb{C}$, the polylogarithm is
    defined by

    .. math:: \\operatorname{Li}_s(z) = \\sum_{n=1}^\\infty \\frac{z^n}{n^s},

    where the standard branch of the argument is used for $n$. It admits
    an analytic continuation which is branched at $z=1$ (notably not on the
    sheet of initial definition), $z=0$ and $z=\\infty$.

    The name polylogarithm comes from the fact that for $s=1$, the
    polylogarithm is related to the ordinary logarithm (see examples), and that

    .. math:: \\operatorname{Li}_{s+1}(z) =
                    \\int_0^z \\frac{\\operatorname{Li}_s(t)}{t} \\mathrm{d}t.

    The polylogarithm is a special case of the Lerch transcendent:

    .. math:: \\operatorname{Li}_{s}(z) = z \\Phi(z, s, 1).

    Examples
    ========

    For $z \\in \\{0, 1, -1\\}$, the polylogarithm is automatically expressed
    using other functions:

    >>> from sympy import polylog
    >>> from sympy.abc import s
    >>> polylog(s, 0)
    0
    >>> polylog(s, 1)
    zeta(s)
    >>> polylog(s, -1)
    -dirichlet_eta(s)

    If $s$ is a negative integer, $0$ or $1$, the polylogarithm can be
    expressed using elementary functions. This can be done using
    ``expand_func()``:

    >>> from sympy import expand_func
    >>> from sympy.abc import z
    >>> expand_func(polylog(1, z))
    -log(1 - z)
    >>> expand_func(polylog(0, z))
    z/(1 - z)

    The derivative with respect to $z$ can be computed in closed form:

    >>> polylog(s, z).diff(z)
    polylog(s - 1, z)/z

    The polylogarithm can be expressed in terms of the lerch transcendent:

    >>> from sympy import lerchphi
    >>> polylog(s, z).rewrite(lerchphi)
    z*lerchphi(z, s, 1)

    See Also
    ========

    zeta, lerchphi

    """

    @classmethod
    def eval(cls, s, z):
        if z.is_number:
            if z is S.One:
                return zeta(s)
            elif z is S.NegativeOne:
                return -dirichlet_eta(s)
            elif z is S.Zero:
                return S.Zero
            elif s == 2:
                dilogtable = _dilogtable()
                if z in dilogtable:
                    return dilogtable[z]
        if z.is_zero:
            return S.Zero
        zone = z.equals(S.One)
        if zone:
            return zeta(s)
        elif zone is False:
            if s is S.Zero:
                return z / (1 - z)
            elif s is S.NegativeOne:
                return z / (1 - z) ** 2
            if s.is_zero:
                return z / (1 - z)
        if z.has(exp_polar, polar_lift) and (zone or (Abs(z) <= S.One) == True):
            return cls(s, unpolarify(z))

    def fdiff(self, argindex=1):
        s, z = self.args
        if argindex == 2:
            return polylog(s - 1, z) / z
        raise ArgumentIndexError

    def _eval_rewrite_as_lerchphi(self, s, z, **kwargs):
        return z * lerchphi(z, s, 1)

    def _eval_expand_func(self, **hints):
        s, z = self.args
        if s == 1:
            return -log(1 - z)
        if s.is_Integer and s <= 0:
            u = Dummy('u')
            start = u / (1 - u)
            for _ in range(-s):
                start = u * start.diff(u)
            return expand_mul(start).subs(u, z)
        return polylog(s, z)

    def _eval_is_zero(self):
        z = self.args[1]
        if z.is_zero:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args
        z0 = z.subs(x, 0)
        if z0 is S.NaN:
            z0 = z.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if z0.is_zero:
            try:
                _, exp = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self
            if exp.is_positive:
                newn = ceiling(n / exp)
                o = Order(x ** n, x)
                r = z._eval_nseries(x, n, logx, cdir).removeO()
                if r is S.Zero:
                    return o
                term = r
                s = [term]
                for k in range(2, newn):
                    term *= r
                    s.append(term / k ** nu)
                return Add(*s) + o
        return super(polylog, self)._eval_nseries(x, n, logx, cdir)