import warnings
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial
from scipy.stats import rv_continuous
import scipy.special as special
Construct the Edgeworth expansion pdf given cumulants.

    Parameters
    ----------
    cum : array_like
        `cum[j]` contains `(j+1)`-th cumulant: cum[0] is the mean,
        cum[1] is the variance and so on.

    Notes
    -----
    This is actually an asymptotic rather than convergent series, hence
    higher orders of the expansion may or may not improve the result.
    In a strongly non-Gaussian case, it is possible that the density
    becomes negative, especially far out in the tails.

    Examples
    --------
    Construct the 4th order expansion for the chi-square distribution using
    the known values of the cumulants:

    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> from scipy.special import factorial
    >>> df = 12
    >>> chi2_c = [2**(j-1) * factorial(j-1) * df for j in range(1, 5)]
    >>> edgw_chi2 = ExpandedNormal(chi2_c, name='edgw_chi2', momtype=0)

    Calculate several moments:
    >>> m, v = edgw_chi2.stats(moments='mv')
    >>> np.allclose([m, v], [df, 2 * df])
    True

    Plot the density function:
    >>> mu, sigma = df, np.sqrt(2*df)
    >>> x = np.linspace(mu - 3*sigma, mu + 3*sigma)
    >>> fig1 = plt.plot(x, stats.chi2.pdf(x, df=df), 'g-', lw=4, alpha=0.5)
    >>> fig2 = plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b--', lw=4, alpha=0.5)
    >>> fig3 = plt.plot(x, edgw_chi2.pdf(x), 'r-', lw=2)
    >>> plt.show()

    References
    ----------
    .. [*] E.A. Cornish and R.A. Fisher, Moments and cumulants in the
         specification of distributions, Revue de l'Institut Internat.
         de Statistique. 5: 307 (1938), reprinted in
         R.A. Fisher, Contributions to Mathematical Statistics. Wiley, 1950.
    .. [*] https://en.wikipedia.org/wiki/Edgeworth_series
    .. [*] S. Blinnikov and R. Moessner, Expansions for nearly Gaussian
        distributions, Astron. Astrophys. Suppl. Ser. 130, 193 (1998)
    