from math import prod
from sympy.core import Add, S, Dummy, expand_func
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and, fuzzy_not
from sympy.core.numbers import Rational, pi, oo, I
from sympy.core.power import Pow
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.combinatorial.factorials import factorial, rf, RisingFactorial
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps

    The multivariate gamma function is a generalization of the gamma function

    .. math::
        \Gamma_p(z) = \pi^{p(p-1)/4}\prod_{k=1}^p \Gamma[z + (1 - k)/2].

    In a special case, ``multigamma(x, 1) = gamma(x)``.

    Examples
    ========

    >>> from sympy import S, multigamma
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> p = Symbol('p', positive=True, integer=True)

    >>> multigamma(x, p)
    pi**(p*(p - 1)/4)*Product(gamma(-_k/2 + x + 1/2), (_k, 1, p))

    Several special values are known:

    >>> multigamma(1, 1)
    1
    >>> multigamma(4, 1)
    6
    >>> multigamma(S(3)/2, 1)
    sqrt(pi)/2

    Writing ``multigamma`` in terms of the ``gamma`` function:

    >>> multigamma(x, 1)
    gamma(x)

    >>> multigamma(x, 2)
    sqrt(pi)*gamma(x)*gamma(x - 1/2)

    >>> multigamma(x, 3)
    pi**(3/2)*gamma(x)*gamma(x - 1)*gamma(x - 1/2)

    Parameters
    ==========

    p : order or dimension of the multivariate gamma function

    See Also
    ========

    gamma, lowergamma, uppergamma, polygamma, loggamma, digamma, trigamma,
    beta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_gamma_function

    