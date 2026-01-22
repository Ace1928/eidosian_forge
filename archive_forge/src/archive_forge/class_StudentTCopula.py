import numpy as np
from scipy import stats
from statsmodels.compat.scipy import multivariate_t
from statsmodels.distributions.copula.copulas import Copula
class StudentTCopula(EllipticalCopula):
    """Student t copula.

    Parameters
    ----------
    corr : scalar or array_like
        Correlation or scatter matrix for the elliptical copula. In the
        bivariate case, ``corr` can be a scalar and is then considered as
        the correlation coefficient. If ``corr`` is None, then the scatter
        matrix is the identity matrix.
    df : float (optional)
        Degrees of freedom of the multivariate t distribution.
    k_dim : int
        Dimension, number of components in the multivariate random variable.

    Notes
    -----
    Elliptical copulas require that copula parameters are set when the
    instance is created. Those parameters currently cannot be provided in the
    call to methods. (This will most likely change in future versions.)
    If non-empty ``args`` are provided in methods, then a ValueError is raised.
    The ``args`` keyword is provided for a consistent interface across
    copulas.

    References
    ----------
    .. [1] Joe, Harry, 2014, Dependence modeling with copulas. CRC press.
        p. 181
    """

    def __init__(self, corr=None, df=None, k_dim=2):
        super().__init__(k_dim=k_dim)
        if corr is None:
            corr = np.eye(k_dim)
        elif k_dim == 2 and np.size(corr) == 1:
            corr = np.array([[1.0, corr], [corr, 1.0]])
        self.df = df
        self.corr = np.asarray(corr)
        self.args = (corr, df)
        self.distr_uv = stats.t(df=df)
        self.distr_mv = multivariate_t(shape=corr, df=df)

    def cdf(self, u, args=()):
        raise NotImplementedError('CDF not available in closed form.')

    def spearmans_rho(self, corr=None):
        """
        Bivariate Spearman's rho based on correlation coefficient.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Spearman's rho that corresponds to pearson correlation in the
        elliptical copula.
        """
        if corr is None:
            corr = self.corr
        if corr.shape == (2, 2):
            corr = corr[0, 1]
        tau = 6 * np.arcsin(corr / 2) / np.pi
        return tau

    def dependence_tail(self, corr=None):
        """
        Bivariate tail dependence parameter.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Lower and upper tail dependence coefficients of the copula with given
        Pearson correlation coefficient.
        """
        if corr is None:
            corr = self.corr
        if corr.shape == (2, 2):
            corr = corr[0, 1]
        df = self.df
        t = -np.sqrt((df + 1) * (1 - corr) / 1 + corr)
        lam = 2 * stats.t.cdf(t, df + 1)
        return (lam, lam)

    def _arg_from_tau(self, tau):
        return self.corr_from_tau(tau)