import numpy as np
from scipy.interpolate import interp1d
def _conf_set(F, alpha=0.05):
    """
    Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the eCDF.

    Parameters
    ----------
    F : array_like
        The empirical distributions
    alpha : float
        Set alpha for a (1 - alpha) % confidence band.

    Notes
    -----
    Based on the DKW inequality.

    .. math:: P \\left( \\sup_x \\left| F(x) - \\hat(F)_n(X) \\right| >
       \\epsilon \\right) \\leq 2e^{-2n\\epsilon^2}

    References
    ----------
    Wasserman, L. 2006. `All of Nonparametric Statistics`. Springer.
    """
    nobs = len(F)
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2 * nobs))
    lower = np.clip(F - epsilon, 0, 1)
    upper = np.clip(F + epsilon, 0, 1)
    return (lower, upper)