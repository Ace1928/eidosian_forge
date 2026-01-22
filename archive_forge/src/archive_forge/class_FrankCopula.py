import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
class FrankCopula(ArchimedeanCopula):
    """Frank copula.

    Dependence is symmetric.

    .. math::

        C_\\theta(\\mathbf{u}) = -\\frac{1}{\\theta} \\log \\left[ 1-
        \\frac{ \\prod_j (1-\\exp(- \\theta u_j)) }{ (1 - \\exp(-\\theta)-1)^{d -
        1} } \\right]

    with :math:`\\theta\\in \\mathbb{R}\\backslash\\{0\\}, \\mathbf{u} \\in [0, 1]^d`.

    """

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfFrank(), args=args, k_dim=k_dim)
        if theta is not None:
            if theta == 0:
                raise ValueError('Theta must be !=0')
        self.theta = theta

    def rvs(self, nobs=1, args=(), random_state=None):
        rng = check_random_state(random_state)
        th, = self._handle_args(args)
        x = rng.random((nobs, self.k_dim))
        v = stats.logser.rvs(1.0 - np.exp(-th), size=(nobs, 1), random_state=rng)
        return -1.0 / th * np.log(1.0 + np.exp(-(-np.log(x) / v)) * (np.exp(-th) - 1.0))

    def pdf(self, u, args=()):
        u = self._handle_u(u)
        th, = self._handle_args(args)
        if u.shape[-1] != 2:
            return super().pdf(u, th)
        g_ = np.exp(-th * np.sum(u, axis=-1)) - 1
        g1 = np.exp(-th) - 1
        num = -th * g1 * (1 + g_)
        aux = np.prod(np.exp(-th * u) - 1, axis=-1) + g1
        den = aux ** 2
        return num / den

    def cdf(self, u, args=()):
        u = self._handle_u(u)
        th, = self._handle_args(args)
        dim = u.shape[-1]
        num = np.prod(1 - np.exp(-th * u), axis=-1)
        den = (1 - np.exp(-th)) ** (dim - 1)
        return -1.0 / th * np.log(1 - num / den)

    def logpdf(self, u, args=()):
        u = self._handle_u(u)
        th, = self._handle_args(args)
        if u.shape[-1] == 2:
            u1, u2 = (u[..., 0], u[..., 1])
            b = 1 - np.exp(-th)
            pdf = np.log(th * b) - th * (u1 + u2)
            pdf -= 2 * np.log(b - (1 - np.exp(-th * u1)) * (1 - np.exp(-th * u2)))
            return pdf
        else:
            return super().logpdf(u, args)

    def cdfcond_2g1(self, u, args=()):
        """Conditional cdf of second component given the value of first.
        """
        u = self._handle_u(u)
        th, = self._handle_args(args)
        if u.shape[-1] == 2:
            u1, u2 = (u[..., 0], u[..., 1])
            cdfc = np.exp(-th * u1)
            cdfc /= np.expm1(-th) / np.expm1(-th * u2) + np.expm1(-th * u1)
            return cdfc
        else:
            raise NotImplementedError('u needs to be bivariate (2 columns)')

    def ppfcond_2g1(self, q, u1, args=()):
        """Conditional pdf of second component given the value of first.
        """
        u1 = np.asarray(u1)
        th, = self._handle_args(args)
        if u1.shape[-1] == 1:
            ppfc = -np.log(1 + np.expm1(-th) / ((1 / q - 1) * np.exp(-th * u1) + 1)) / th
            return ppfc
        else:
            raise NotImplementedError('u needs to be bivariate (2 columns)')

    def tau(self, theta=None):
        if theta is None:
            theta = self.theta
        return tau_frank(theta)

    def theta_from_tau(self, tau):
        MIN_FLOAT_LOG = np.log(sys.float_info.min)
        MAX_FLOAT_LOG = np.log(sys.float_info.max)

        def _theta_from_tau(alpha):
            return self.tau(theta=alpha) - tau
        start = 0.5 if tau < 0.11 else 2
        result = optimize.least_squares(_theta_from_tau, start, bounds=(MIN_FLOAT_LOG, MAX_FLOAT_LOG))
        theta = result.x[0]
        return theta