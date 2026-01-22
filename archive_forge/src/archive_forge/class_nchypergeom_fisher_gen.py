from functools import partial
from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d
from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
import numpy as np
from ._distn_infrastructure import (rv_discrete, get_distribution_names,
import scipy.stats._boost as _boost
from ._biasedurn import (_PyFishersNCHypergeometric,
class nchypergeom_fisher_gen(_nchypergeom_gen):
    """A Fisher's noncentral hypergeometric discrete random variable.

    Fisher's noncentral hypergeometric distribution models drawing objects of
    two types from a bin. `M` is the total number of objects, `n` is the
    number of Type I objects, and `odds` is the odds ratio: the odds of
    selecting a Type I object rather than a Type II object when there is only
    one object of each type.
    The random variate represents the number of Type I objects drawn if we
    take a handful of objects from the bin at once and find out afterwards
    that we took `N` objects.

    %(before_notes)s

    See Also
    --------
    nchypergeom_wallenius, hypergeom, nhypergeom

    Notes
    -----
    Let mathematical symbols :math:`N`, :math:`n`, and :math:`M` correspond
    with parameters `N`, `n`, and `M` (respectively) as defined above.

    The probability mass function is defined as

    .. math::

        p(x; M, n, N, \\omega) =
        \\frac{\\binom{n}{x}\\binom{M - n}{N-x}\\omega^x}{P_0},

    for
    :math:`x \\in [x_l, x_u]`,
    :math:`M \\in {\\mathbb N}`,
    :math:`n \\in [0, M]`,
    :math:`N \\in [0, M]`,
    :math:`\\omega > 0`,
    where
    :math:`x_l = \\max(0, N - (M - n))`,
    :math:`x_u = \\min(N, n)`,

    .. math::

        P_0 = \\sum_{y=x_l}^{x_u} \\binom{n}{y}\\binom{M - n}{N-y}\\omega^y,

    and the binomial coefficients are defined as

    .. math:: \\binom{n}{k} \\equiv \\frac{n!}{k! (n - k)!}.

    `nchypergeom_fisher` uses the BiasedUrn package by Agner Fog with
    permission for it to be distributed under SciPy's license.

    The symbols used to denote the shape parameters (`N`, `n`, and `M`) are not
    universally accepted; they are chosen for consistency with `hypergeom`.

    Note that Fisher's noncentral hypergeometric distribution is distinct
    from Wallenius' noncentral hypergeometric distribution, which models
    drawing a pre-determined `N` objects from a bin one by one.
    When the odds ratio is unity, however, both distributions reduce to the
    ordinary hypergeometric distribution.

    %(after_notes)s

    References
    ----------
    .. [1] Agner Fog, "Biased Urn Theory".
           https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf

    .. [2] "Fisher's noncentral hypergeometric distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution

    %(example)s

    """
    rvs_name = 'rvs_fisher'
    dist = _PyFishersNCHypergeometric