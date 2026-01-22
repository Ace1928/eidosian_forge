from sympy.core import S
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate

    The Generalized Regularized Incomplete Beta function is given by

    .. math::
        \mathrm{I}_{(x_1, x_2)}(a, b) = \frac{\mathrm{B}_{(x_1, x_2)}(a, b)}{\mathrm{B}(a, b)}

    The Regularized Incomplete Beta function is a special case
    of the Generalized Regularized Incomplete Beta function :

    .. math:: \mathrm{I}_z (a, b) = \mathrm{I}_{(0, z)}(a, b)

    The Regularized Incomplete Beta function is the cumulative distribution
    function of the beta distribution.

    Examples
    ========

    >>> from sympy import betainc_regularized, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Regularized Incomplete Beta
    function is given by:

    >>> betainc_regularized(a, b, x1, x2)
    betainc_regularized(a, b, x1, x2)

    The Regularized Incomplete Beta function
    can be obtained as follows:

    >>> betainc_regularized(a, b, 0, x)
    betainc_regularized(a, b, 0, x)

    The Regularized Incomplete Beta function
    obeys the mirror symmetry:

    >>> conjugate(betainc_regularized(a, b, x1, x2))
    betainc_regularized(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Regularized Incomplete Beta function
    to arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc_regularized, pi, E
    >>> betainc_regularized(1, 2, 0, 0.25).evalf(10)
    0.4375000000
    >>> betainc_regularized(pi, E, 0, 1).evalf(5)
    1.00000

    The Generalized Regularized Incomplete Beta function can be
    expressed in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc_regularized(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/(a*beta(a, b))

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

    