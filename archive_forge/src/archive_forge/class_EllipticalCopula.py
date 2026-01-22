import numpy as np
from scipy import stats
from statsmodels.compat.scipy import multivariate_t
from statsmodels.distributions.copula.copulas import Copula
class EllipticalCopula(Copula):
    """Base class for elliptical copula

    This class requires subclassing and currently does not have generic
    methods based on an elliptical generator.

    Notes
    -----
    Elliptical copulas require that copula parameters are set when the
    instance is created. Those parameters currently cannot be provided in the
    call to methods. (This will most likely change in future versions.)
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    """

    def _handle_args(self, args):
        if args != () and args is not None:
            msg = 'Methods in elliptical copulas use copula parameters in attributes. `arg` in the method is ignored'
            raise ValueError(msg)
        else:
            return args

    def rvs(self, nobs=1, args=(), random_state=None):
        self._handle_args(args)
        x = self.distr_mv.rvs(size=nobs, random_state=random_state)
        return self.distr_uv.cdf(x)

    def pdf(self, u, args=()):
        self._handle_args(args)
        ppf = self.distr_uv.ppf(u)
        mv_pdf_ppf = self.distr_mv.pdf(ppf)
        return mv_pdf_ppf / np.prod(self.distr_uv.pdf(ppf), axis=-1)

    def cdf(self, u, args=()):
        self._handle_args(args)
        ppf = self.distr_uv.ppf(u)
        return self.distr_mv.cdf(ppf)

    def tau(self, corr=None):
        """Bivariate kendall's tau based on correlation coefficient.

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Kendall's tau that corresponds to pearson correlation in the
        elliptical copula.
        """
        if corr is None:
            corr = self.corr
        if corr.shape == (2, 2):
            corr = corr[0, 1]
        rho = 2 * np.arcsin(corr) / np.pi
        return rho

    def corr_from_tau(self, tau):
        """Pearson correlation from kendall's tau.

        Parameters
        ----------
        tau : array_like
            Kendall's tau correlation coefficient.

        Returns
        -------
        Pearson correlation coefficient for given tau in elliptical
        copula. This can be used as parameter for an elliptical copula.
        """
        corr = np.sin(tau * np.pi / 2)
        return corr

    def fit_corr_param(self, data):
        """Copula correlation parameter using Kendall's tau of sample data.

        Parameters
        ----------
        data : array_like
            Sample data used to fit `theta` using Kendall's tau.

        Returns
        -------
        corr_param : float
            Correlation parameter of the copula, ``theta`` in Archimedean and
            pearson correlation in elliptical.
            If k_dim > 2, then average tau is used.
        """
        x = np.asarray(data)
        if x.shape[1] == 2:
            tau = stats.kendalltau(x[:, 0], x[:, 1])[0]
        else:
            k = self.k_dim
            tau = np.eye(k)
            for i in range(k):
                for j in range(i + 1, k):
                    tau_ij = stats.kendalltau(x[..., i], x[..., j])[0]
                    tau[i, j] = tau[j, i] = tau_ij
        return self._arg_from_tau(tau)