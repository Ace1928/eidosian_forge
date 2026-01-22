import numpy as np
from scipy.stats import norm as Gaussian
from statsmodels.tools import tools
from statsmodels.tools.validation import array_like, float_like
from . import norms
from ._qn import _qn
class HuberScale:
    """
    Huber's scaling for fitting robust linear models.

    Huber's scale is intended to be used as the scale estimate in the
    IRLS algorithm and is slightly different than the `Huber` class.

    Parameters
    ----------
    d : float, optional
        d is the tuning constant for Huber's scale.  Default is 2.5
    tol : float, optional
        The convergence tolerance
    maxiter : int, optiona
        The maximum number of iterations.  The default is 30.

    Methods
    -------
    call
        Return's Huber's scale computed as below

    Notes
    -----
    Huber's scale is the iterative solution to

    scale_(i+1)**2 = 1/(n*h)*sum(chi(r/sigma_i)*sigma_i**2

    where the Huber function is

    chi(x) = (x**2)/2       for \\|x\\| < d
    chi(x) = (d**2)/2       for \\|x\\| >= d

    and the Huber constant h = (n-p)/n*(d**2 + (1-d**2)*
    scipy.stats.norm.cdf(d) - .5 - d*sqrt(2*pi)*exp(-0.5*d**2)
    """

    def __init__(self, d=2.5, tol=1e-08, maxiter=30):
        self.d = d
        self.tol = tol
        self.maxiter = maxiter

    def __call__(self, df_resid, nobs, resid):
        h = df_resid / nobs * (self.d ** 2 + (1 - self.d ** 2) * Gaussian.cdf(self.d) - 0.5 - self.d / np.sqrt(2 * np.pi) * np.exp(-0.5 * self.d ** 2))
        s = mad(resid)

        def subset(x):
            return np.less(np.abs(resid / x), self.d)

        def chi(s):
            return subset(s) * (resid / s) ** 2 / 2 + (1 - subset(s)) * (self.d ** 2 / 2)
        scalehist = [np.inf, s]
        niter = 1
        while np.abs(scalehist[niter - 1] - scalehist[niter]) > self.tol and niter < self.maxiter:
            nscale = np.sqrt(1 / (nobs * h) * np.sum(chi(scalehist[-1])) * scalehist[-1] ** 2)
            scalehist.append(nscale)
            niter += 1
        return scalehist[-1]