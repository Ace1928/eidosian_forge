import numpy as np
from scipy.stats import norm as Gaussian
from statsmodels.tools import tools
from statsmodels.tools.validation import array_like, float_like
from . import norms
from ._qn import _qn
def _estimate_both(self, a, scale, mu, axis, est_mu, n):
    """
        Estimate scale and location simultaneously with the following
        pseudo_loop:

        while not_converged:
            mu, scale = estimate_location(a, scale, mu), estimate_scale(a, scale, mu)

        where estimate_location is an M-estimator and estimate_scale implements
        the check used in Section 5.5 of Venables & Ripley
        """
    for _ in range(self.maxiter):
        if est_mu:
            if self.norm is None:
                nmu = np.clip(a, mu - self.c * scale, mu + self.c * scale).sum(axis) / a.shape[axis]
            else:
                nmu = norms.estimate_location(a, scale, self.norm, axis, mu, self.maxiter, self.tol)
        else:
            nmu = mu.squeeze()
        nmu = tools.unsqueeze(nmu, axis, a.shape)
        subset = np.less_equal(np.abs((a - mu) / scale), self.c)
        card = subset.sum(axis)
        scale_num = np.sum(subset * (a - nmu) ** 2, axis)
        scale_denom = n * self.gamma - (a.shape[axis] - card) * self.c ** 2
        nscale = np.sqrt(scale_num / scale_denom)
        nscale = tools.unsqueeze(nscale, axis, a.shape)
        test1 = np.all(np.less_equal(np.abs(scale - nscale), nscale * self.tol))
        test2 = np.all(np.less_equal(np.abs(mu - nmu), nscale * self.tol))
        if not (test1 and test2):
            mu = nmu
            scale = nscale
        else:
            return (nmu.squeeze(), nscale.squeeze())
    raise ValueError('joint estimation of location and scale failed to converge in %d iterations' % self.maxiter)