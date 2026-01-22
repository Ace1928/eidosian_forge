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
class meijerg(TupleParametersBase):
    """
    The Meijer G-function is defined by a Mellin-Barnes type integral that
    resembles an inverse Mellin transform. It generalizes the hypergeometric
    functions.

    Explanation
    ===========

    The Meijer G-function depends on four sets of parameters. There are
    "*numerator parameters*"
    $a_1, \\ldots, a_n$ and $a_{n+1}, \\ldots, a_p$, and there are
    "*denominator parameters*"
    $b_1, \\ldots, b_m$ and $b_{m+1}, \\ldots, b_q$.
    Confusingly, it is traditionally denoted as follows (note the position
    of $m$, $n$, $p$, $q$, and how they relate to the lengths of the four
    parameter vectors):

    .. math ::
        G_{p,q}^{m,n} \\left(\\begin{matrix}a_1, \\cdots, a_n & a_{n+1}, \\cdots, a_p \\\\
                                        b_1, \\cdots, b_m & b_{m+1}, \\cdots, b_q
                          \\end{matrix} \\middle| z \\right).

    However, in SymPy the four parameter vectors are always available
    separately (see examples), so that there is no need to keep track of the
    decorating sub- and super-scripts on the G symbol.

    The G function is defined as the following integral:

    .. math ::
         \\frac{1}{2 \\pi i} \\int_L \\frac{\\prod_{j=1}^m \\Gamma(b_j - s)
         \\prod_{j=1}^n \\Gamma(1 - a_j + s)}{\\prod_{j=m+1}^q \\Gamma(1- b_j +s)
         \\prod_{j=n+1}^p \\Gamma(a_j - s)} z^s \\mathrm{d}s,

    where $\\Gamma(z)$ is the gamma function. There are three possible
    contours which we will not describe in detail here (see the references).
    If the integral converges along more than one of them, the definitions
    agree. The contours all separate the poles of $\\Gamma(1-a_j+s)$
    from the poles of $\\Gamma(b_k-s)$, so in particular the G function
    is undefined if $a_j - b_k \\in \\mathbb{Z}_{>0}$ for some
    $j \\le n$ and $k \\le m$.

    The conditions under which one of the contours yields a convergent integral
    are complicated and we do not state them here, see the references.

    Please note currently the Meijer G-function constructor does *not* check any
    convergence conditions.

    Examples
    ========

    You can pass the parameters either as four separate vectors:

    >>> from sympy import meijerg, Tuple, pprint
    >>> from sympy.abc import x, a
    >>> pprint(meijerg((1, 2), (a, 4), (5,), [], x), use_unicode=False)
     __1, 2 /1, 2  a, 4 |  \\
    /__     |           | x|
    \\_|4, 1 \\ 5         |  /

    Or as two nested vectors:

    >>> pprint(meijerg([(1, 2), (3, 4)], ([5], Tuple()), x), use_unicode=False)
     __1, 2 /1, 2  3, 4 |  \\
    /__     |           | x|
    \\_|4, 1 \\ 5         |  /

    As with the hypergeometric function, the parameters may be passed as
    arbitrary iterables. Vectors of length zero and one also have to be
    passed as iterables. The parameters need not be constants, but if they
    depend on the argument then not much implemented functionality should be
    expected.

    All the subvectors of parameters are available:

    >>> from sympy import pprint
    >>> g = meijerg([1], [2], [3], [4], x)
    >>> pprint(g, use_unicode=False)
     __1, 1 /1  2 |  \\
    /__     |     | x|
    \\_|2, 2 \\3  4 |  /
    >>> g.an
    (1,)
    >>> g.ap
    (1, 2)
    >>> g.aother
    (2,)
    >>> g.bm
    (3,)
    >>> g.bq
    (3, 4)
    >>> g.bother
    (4,)

    The Meijer G-function generalizes the hypergeometric functions.
    In some cases it can be expressed in terms of hypergeometric functions,
    using Slater's theorem. For example:

    >>> from sympy import hyperexpand
    >>> from sympy.abc import a, b, c
    >>> hyperexpand(meijerg([a], [], [c], [b], x), allow_hyper=True)
    x**c*gamma(-a + c + 1)*hyper((-a + c + 1,),
                                 (-b + c + 1,), -x)/gamma(-b + c + 1)

    Thus the Meijer G-function also subsumes many named functions as special
    cases. You can use ``expand_func()`` or ``hyperexpand()`` to (try to)
    rewrite a Meijer G-function in terms of named special functions. For
    example:

    >>> from sympy import expand_func, S
    >>> expand_func(meijerg([[],[]], [[0],[]], -x))
    exp(x)
    >>> hyperexpand(meijerg([[],[]], [[S(1)/2],[0]], (x/2)**2))
    sin(x)/sqrt(pi)

    See Also
    ========

    hyper
    sympy.simplify.hyperexpand

    References
    ==========

    .. [1] Luke, Y. L. (1969), The Special Functions and Their Approximations,
           Volume 1
    .. [2] https://en.wikipedia.org/wiki/Meijer_G-function

    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 5:
            args = [(args[0], args[1]), (args[2], args[3]), args[4]]
        if len(args) != 3:
            raise TypeError("args must be either as, as', bs, bs', z or as, bs, z")

        def tr(p):
            if len(p) != 2:
                raise TypeError('wrong argument')
            return TupleArg(_prep_tuple(p[0]), _prep_tuple(p[1]))
        arg0, arg1 = (tr(args[0]), tr(args[1]))
        if Tuple(arg0, arg1).has(oo, zoo, -oo):
            raise ValueError('G-function parameters must be finite')
        if any(((a - b).is_Integer and a - b > 0 for a in arg0[0] for b in arg1[0])):
            raise ValueError('no parameter a1, ..., an may differ from any b1, ..., bm by a positive integer')
        return Function.__new__(cls, arg0, arg1, args[2], **kwargs)

    def fdiff(self, argindex=3):
        if argindex != 3:
            return self._diff_wrt_parameter(argindex[1])
        if len(self.an) >= 1:
            a = list(self.an)
            a[0] -= 1
            G = meijerg(a, self.aother, self.bm, self.bother, self.argument)
            return 1 / self.argument * ((self.an[0] - 1) * self + G)
        elif len(self.bm) >= 1:
            b = list(self.bm)
            b[0] += 1
            G = meijerg(self.an, self.aother, b, self.bother, self.argument)
            return 1 / self.argument * (self.bm[0] * self - G)
        else:
            return S.Zero

    def _diff_wrt_parameter(self, idx):
        an = list(self.an)
        ap = list(self.aother)
        bm = list(self.bm)
        bq = list(self.bother)
        if idx < len(an):
            an.pop(idx)
        else:
            idx -= len(an)
            if idx < len(ap):
                ap.pop(idx)
            else:
                idx -= len(ap)
                if idx < len(bm):
                    bm.pop(idx)
                else:
                    bq.pop(idx - len(bm))
        pairs1 = []
        pairs2 = []
        for l1, l2, pairs in [(an, bq, pairs1), (ap, bm, pairs2)]:
            while l1:
                x = l1.pop()
                found = None
                for i, y in enumerate(l2):
                    if not Mod((x - y).simplify(), 1):
                        found = i
                        break
                if found is None:
                    raise NotImplementedError('Derivative not expressible as G-function?')
                y = l2[i]
                l2.pop(i)
                pairs.append((x, y))
        res = log(self.argument) * self
        for a, b in pairs1:
            sign = 1
            n = a - b
            base = b
            if n < 0:
                sign = -1
                n = b - a
                base = a
            for k in range(n):
                res -= sign * meijerg(self.an + (base + k + 1,), self.aother, self.bm, self.bother + (base + k + 0,), self.argument)
        for a, b in pairs2:
            sign = 1
            n = b - a
            base = a
            if n < 0:
                sign = -1
                n = a - b
                base = b
            for k in range(n):
                res -= sign * meijerg(self.an, self.aother + (base + k + 1,), self.bm + (base + k + 0,), self.bother, self.argument)
        return res

    def get_period(self):
        """
        Return a number $P$ such that $G(x*exp(I*P)) == G(x)$.

        Examples
        ========

        >>> from sympy import meijerg, pi, S
        >>> from sympy.abc import z

        >>> meijerg([1], [], [], [], z).get_period()
        2*pi
        >>> meijerg([pi], [], [], [], z).get_period()
        oo
        >>> meijerg([1, 2], [], [], [], z).get_period()
        oo
        >>> meijerg([1,1], [2], [1, S(1)/2, S(1)/3], [1], z).get_period()
        12*pi

        """

        def compute(l):
            for i, b in enumerate(l):
                if not b.is_Rational:
                    return oo
                for j in range(i + 1, len(l)):
                    if not Mod((b - l[j]).simplify(), 1):
                        return oo
            return reduce(ilcm, (x.q for x in l), 1)
        beta = compute(self.bm)
        alpha = compute(self.an)
        p, q = (len(self.ap), len(self.bq))
        if p == q:
            if oo in (alpha, beta):
                return oo
            return 2 * pi * ilcm(alpha, beta)
        elif p < q:
            return 2 * pi * beta
        else:
            return 2 * pi * alpha

    def _eval_expand_func(self, **hints):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self)

    def _eval_evalf(self, prec):
        import mpmath
        znum = self.argument._eval_evalf(prec)
        if znum.has(exp_polar):
            znum, branch = znum.as_coeff_mul(exp_polar)
            if len(branch) != 1:
                return
            branch = branch[0].args[0] / I
        else:
            branch = S.Zero
        n = ceiling(abs(branch / pi)) + 1
        znum = znum ** (S.One / n) * exp(I * branch / n)
        try:
            [z, r, ap, bq] = [arg._to_mpmath(prec) for arg in [znum, 1 / n, self.args[0], self.args[1]]]
        except ValueError:
            return
        with mpmath.workprec(prec):
            v = mpmath.meijerg(ap, bq, z, r)
        return Expr._from_mpmath(v, prec)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.simplify.hyperexpand import hyperexpand
        return hyperexpand(self).as_leading_term(x, logx=logx, cdir=cdir)

    def integrand(self, s):
        """ Get the defining integrand D(s). """
        from sympy.functions.special.gamma_functions import gamma
        return self.argument ** s * Mul(*(gamma(b - s) for b in self.bm)) * Mul(*(gamma(1 - a + s) for a in self.an)) / Mul(*(gamma(1 - b + s) for b in self.bother)) / Mul(*(gamma(a - s) for a in self.aother))

    @property
    def argument(self):
        """ Argument of the Meijer G-function. """
        return self.args[2]

    @property
    def an(self):
        """ First set of numerator parameters. """
        return Tuple(*self.args[0][0])

    @property
    def ap(self):
        """ Combined numerator parameters. """
        return Tuple(*self.args[0][0] + self.args[0][1])

    @property
    def aother(self):
        """ Second set of numerator parameters. """
        return Tuple(*self.args[0][1])

    @property
    def bm(self):
        """ First set of denominator parameters. """
        return Tuple(*self.args[1][0])

    @property
    def bq(self):
        """ Combined denominator parameters. """
        return Tuple(*self.args[1][0] + self.args[1][1])

    @property
    def bother(self):
        """ Second set of denominator parameters. """
        return Tuple(*self.args[1][1])

    @property
    def _diffargs(self):
        return self.ap + self.bq

    @property
    def nu(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        return sum(self.bq) - sum(self.ap)

    @property
    def delta(self):
        """ A quantity related to the convergence region of the integral,
            c.f. references. """
        return len(self.bm) + len(self.an) - S(len(self.ap) + len(self.bq)) / 2

    @property
    def is_number(self):
        """ Returns true if expression has numeric data only. """
        return not self.free_symbols