from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
class hyper(TupleParametersBase):
    """
    The generalized hypergeometric function is defined by a series where
    the ratios of successive terms are a rational function of the summation
    index. When convergent, it is continued analytically to the largest
    possible domain.

    Explanation
    ===========

    The hypergeometric function depends on two vectors of parameters, called
    the numerator parameters $a_p$, and the denominator parameters
    $b_q$. It also has an argument $z$. The series definition is

    .. math ::
        {}_pF_q\\left(\\begin{matrix} a_1, \\cdots, a_p \\\\ b_1, \\cdots, b_q \\end{matrix}
                     \\middle| z \\right)
        = \\sum_{n=0}^\\infty \\frac{(a_1)_n \\cdots (a_p)_n}{(b_1)_n \\cdots (b_q)_n}
                            \\frac{z^n}{n!},

    where $(a)_n = (a)(a+1)\\cdots(a+n-1)$ denotes the rising factorial.

    If one of the $b_q$ is a non-positive integer then the series is
    undefined unless one of the $a_p$ is a larger (i.e., smaller in
    magnitude) non-positive integer. If none of the $b_q$ is a
    non-positive integer and one of the $a_p$ is a non-positive
    integer, then the series reduces to a polynomial. To simplify the
    following discussion, we assume that none of the $a_p$ or
    $b_q$ is a non-positive integer. For more details, see the
    references.

    The series converges for all $z$ if $p \\le q$, and thus
    defines an entire single-valued function in this case. If $p =
    q+1$ the series converges for $|z| < 1$, and can be continued
    analytically into a half-plane. If $p > q+1$ the series is
    divergent for all $z$.

    Please note the hypergeometric function constructor currently does *not*
    check if the parameters actually yield a well-defined function.

    Examples
    ========

    The parameters $a_p$ and $b_q$ can be passed as arbitrary
    iterables, for example:

    >>> from sympy import hyper
    >>> from sympy.abc import x, n, a
    >>> hyper((1, 2, 3), [3, 4], x)
    hyper((1, 2, 3), (3, 4), x)

    There is also pretty printing (it looks better using Unicode):

    >>> from sympy import pprint
    >>> pprint(hyper((1, 2, 3), [3, 4], x), use_unicode=False)
      _
     |_  /1, 2, 3 |  \\
     |   |        | x|
    3  2 \\  3, 4  |  /

    The parameters must always be iterables, even if they are vectors of
    length one or zero:

    >>> hyper((1, ), [], x)
    hyper((1,), (), x)

    But of course they may be variables (but if they depend on $x$ then you
    should not expect much implemented functionality):

    >>> hyper((n, a), (n**2,), x)
    hyper((n, a), (n**2,), x)

    The hypergeometric function generalizes many named special functions.
    The function ``hyperexpand()`` tries to express a hypergeometric function
    using named special functions. For example:

    >>> from sympy import hyperexpand
    >>> hyperexpand(hyper([], [], x))
    exp(x)

    You can also use ``expand_func()``:

    >>> from sympy import expand_func
    >>> expand_func(x*hyper([1, 1], [2], -x))
    log(x + 1)

    More examples:

    >>> from sympy import S
    >>> hyperexpand(hyper([], [S(1)/2], -x**2/4))
    cos(x)
    >>> hyperexpand(x*hyper([S(1)/2, S(1)/2], [S(3)/2], x**2))
    asin(x)

    We can also sometimes ``hyperexpand()`` parametric functions:

    >>> from sympy.abc import a
    >>> hyperexpand(hyper([-a], [], x))
    (1 - x)**a

    See Also
    ========

    sympy.simplify.hyperexpand
    gamma
    meijerg

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Generalized_hypergeometric_function

    """

    def __new__(cls, ap, bq, z, **kwargs):
        return Function.__new__(cls, _prep_tuple(ap), _prep_tuple(bq), z, **kwargs)

    @classmethod
    def eval(cls, ap, bq, z):
        if len(ap) <= len(bq) or (len(ap) == len(bq) + 1 and (Abs(z) <= 1) == True):
            nz = unpolarify(z)
            if z != nz:
                return hyper(ap, bq, nz)

    def fdiff(self, argindex=3):
        if argindex != 3:
            raise ArgumentIndexError(self, argindex)
        nap = Tuple(*[a + 1 for a in self.ap])
        nbq = Tuple(*[b + 1 for b in self.bq])
        fac = Mul(*self.ap) / Mul(*self.bq)
        return fac * hyper(nap, nbq, self.argument)

    def _eval_expand_func(self, **hints):
        from sympy.functions.special.gamma_functions import gamma
        from sympy.simplify.hyperexpand import hyperexpand
        if len(self.ap) == 2 and len(self.bq) == 1 and (self.argument == 1):
            a, b = self.ap
            c = self.bq[0]
            return gamma(c) * gamma(c - a - b) / gamma(c - a) / gamma(c - b)
        return hyperexpand(self)

    def _eval_rewrite_as_Sum(self, ap, bq, z, **kwargs):
        from sympy.concrete.summations import Sum
        n = Dummy('n', integer=True)
        rfap = [RisingFactorial(a, n) for a in ap]
        rfbq = [RisingFactorial(b, n) for b in bq]
        coeff = Mul(*rfap) / Mul(*rfbq)
        return Piecewise((Sum(coeff * z ** n / factorial(n), (n, 0, oo)), self.convergence_statement), (self, True))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[2]
        x0 = arg.subs(x, 0)
        if x0 is S.NaN:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 is S.Zero:
            return S.One
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        arg = self.args[2]
        x0 = arg.limit(x, 0)
        ap = self.args[0]
        bq = self.args[1]
        if x0 != 0:
            return super()._eval_nseries(x, n, logx)
        terms = []
        for i in range(n):
            num = Mul(*[RisingFactorial(a, i) for a in ap])
            den = Mul(*[RisingFactorial(b, i) for b in bq])
            terms.append(num / den * arg ** i / factorial(i))
        return Add(*terms) + Order(x ** n, x)

    @property
    def argument(self):
        """ Argument of the hypergeometric function. """
        return self.args[2]

    @property
    def ap(self):
        """ Numerator parameters of the hypergeometric function. """
        return Tuple(*self.args[0])

    @property
    def bq(self):
        """ Denominator parameters of the hypergeometric function. """
        return Tuple(*self.args[1])

    @property
    def _diffargs(self):
        return self.ap + self.bq

    @property
    def eta(self):
        """ A quantity related to the convergence of the series. """
        return sum(self.ap) - sum(self.bq)

    @property
    def radius_of_convergence(self):
        """
        Compute the radius of convergence of the defining series.

        Explanation
        ===========

        Note that even if this is not ``oo``, the function may still be
        evaluated outside of the radius of convergence by analytic
        continuation. But if this is zero, then the function is not actually
        defined anywhere else.

        Examples
        ========

        >>> from sympy import hyper
        >>> from sympy.abc import z
        >>> hyper((1, 2), [3], z).radius_of_convergence
        1
        >>> hyper((1, 2, 3), [4], z).radius_of_convergence
        0
        >>> hyper((1, 2), (3, 4), z).radius_of_convergence
        oo

        """
        if any((a.is_integer and (a <= 0) == True for a in self.ap + self.bq)):
            aints = [a for a in self.ap if a.is_Integer and (a <= 0) == True]
            bints = [a for a in self.bq if a.is_Integer and (a <= 0) == True]
            if len(aints) < len(bints):
                return S.Zero
            popped = False
            for b in bints:
                cancelled = False
                while aints:
                    a = aints.pop()
                    if a >= b:
                        cancelled = True
                        break
                    popped = True
                if not cancelled:
                    return S.Zero
            if aints or popped:
                return oo
        if len(self.ap) == len(self.bq) + 1:
            return S.One
        elif len(self.ap) <= len(self.bq):
            return oo
        else:
            return S.Zero

    @property
    def convergence_statement(self):
        """ Return a condition on z under which the series converges. """
        R = self.radius_of_convergence
        if R == 0:
            return False
        if R == oo:
            return True
        e = self.eta
        z = self.argument
        c1 = And(re(e) < 0, abs(z) <= 1)
        c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
        c3 = And(re(e) >= 1, abs(z) < 1)
        return Or(c1, c2, c3)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)