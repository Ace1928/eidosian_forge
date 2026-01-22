from sympy.core import S
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
class betainc(Function):
    """
    The Generalized Incomplete Beta function is defined as

    .. math::
        \\mathrm{B}_{(x_1, x_2)}(a, b) = \\int_{x_1}^{x_2} t^{a - 1} (1 - t)^{b - 1} dt

    The Incomplete Beta function is a special case
    of the Generalized Incomplete Beta function :

    .. math:: \\mathrm{B}_z (a, b) = \\mathrm{B}_{(0, z)}(a, b)

    The Incomplete Beta function satisfies :

    .. math:: \\mathrm{B}_z (a, b) = (-1)^a \\mathrm{B}_{\\frac{z}{z - 1}} (a, 1 - a - b)

    The Beta function is a special case of the Incomplete Beta function :

    .. math:: \\mathrm{B}(a, b) = \\mathrm{B}_{1}(a, b)

    Examples
    ========

    >>> from sympy import betainc, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Incomplete Beta function is given by:

    >>> betainc(a, b, x1, x2)
    betainc(a, b, x1, x2)

    The Incomplete Beta function can be obtained as follows:

    >>> betainc(a, b, 0, x)
    betainc(a, b, 0, x)

    The Incomplete Beta function obeys the mirror symmetry:

    >>> conjugate(betainc(a, b, x1, x2))
    betainc(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Incomplete Beta function to
    arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc, I
    >>> betainc(2, 3, 4, 5).evalf(10)
    56.08333333
    >>> betainc(0.75, 1 - 4*I, 0, 2 + 3*I).evalf(25)
    0.2241657956955709603655887 + 0.3619619242700451992411724*I

    The Generalized Incomplete Beta function can be expressed
    in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/a

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            return -(1 - x1) ** (b - 1) * x1 ** (a - 1)
        elif argindex == 4:
            return (1 - x2) ** (b - 1) * x2 ** (a - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_mpmath(self):
        return (betainc_mpmath_fix, self.args)

    def _eval_is_real(self):
        if all((arg.is_real for arg in self.args)):
            return True

    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))

    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy('t')
        return Integral(t ** (a - 1) * (1 - t) ** (b - 1), (t, x1, x2))

    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        return (x2 ** a * hyper((a, 1 - b), (a + 1,), x2) - x1 ** a * hyper((a, 1 - b), (a + 1,), x1)) / a