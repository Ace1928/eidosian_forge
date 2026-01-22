import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
class GumbelCopula(ArchimedeanCopula):
    """Gumbel copula.

    Dependence is greater in the positive tail than in the negative.

    .. math::

        C_\\theta(u,v) = \\exp\\!\\left[ -\\left( (-\\log(u))^\\theta +
        (-\\log(v))^\\theta \\right)^{1/\\theta} \\right]

    with :math:`\\theta\\in[1,\\infty)`.

    """

    def __init__(self, theta=None, k_dim=2):
        if theta is not None:
            args = (theta,)
        else:
            args = ()
        super().__init__(transforms.TransfGumbel(), args=args, k_dim=k_dim)
        if theta is not None:
            if theta <= 1:
                raise ValueError('Theta must be > 1')
        self.theta = theta

    def rvs(self, nobs=1, args=(), random_state=None):
        rng = check_random_state(random_state)
        th, = self._handle_args(args)
        x = rng.random((nobs, self.k_dim))
        v = stats.levy_stable.rvs(1.0 / th, 1.0, 0, np.cos(np.pi / (2 * th)) ** th, size=(nobs, 1), random_state=rng)
        if self.k_dim != 2:
            rv = np.exp(-(-np.log(x) / v) ** (1.0 / th))
        else:
            rv = self.transform.inverse(-np.log(x) / v, th)
        return rv

    def pdf(self, u, args=()):
        u = self._handle_u(u)
        th, = self._handle_args(args)
        if u.shape[-1] == 2:
            xy = -np.log(u)
            xy_theta = xy ** th
            sum_xy_theta = np.sum(xy_theta, axis=-1)
            sum_xy_theta_theta = sum_xy_theta ** (1.0 / th)
            a = np.exp(-sum_xy_theta_theta)
            b = sum_xy_theta_theta + th - 1.0
            c = sum_xy_theta ** (1.0 / th - 2)
            d = np.prod(xy, axis=-1) ** (th - 1.0)
            e = np.prod(u, axis=-1) ** (-1.0)
            return a * b * c * d * e
        else:
            return super().pdf(u, args)

    def cdf(self, u, args=()):
        u = self._handle_u(u)
        th, = self._handle_args(args)
        h = np.sum((-np.log(u)) ** th, axis=-1)
        cdf = np.exp(-h ** (1.0 / th))
        return cdf

    def logpdf(self, u, args=()):
        return super().logpdf(u, args=args)

    def tau(self, theta=None):
        if theta is None:
            theta = self.theta
        return (theta - 1) / theta

    def theta_from_tau(self, tau):
        return 1 / (1 - tau)