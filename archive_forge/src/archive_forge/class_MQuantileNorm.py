import numpy as np
class MQuantileNorm(RobustNorm):
    """M-quantiles objective function based on a base norm

    This norm has the same asymmetric structure as the objective function
    in QuantileRegression but replaces the L1 absolute value by a chosen
    base norm.

        rho_q(u) = abs(q - I(q < 0)) * rho_base(u)

    or, equivalently,

        rho_q(u) = q * rho_base(u)  if u >= 0
        rho_q(u) = (1 - q) * rho_base(u)  if u < 0


    Parameters
    ----------
    q : float
        M-quantile, must be between 0 and 1
    base_norm : RobustNorm instance
        basic norm that is transformed into an asymmetric M-quantile norm

    Notes
    -----
    This is mainly for base norms that are not redescending, like HuberT or
    LeastSquares. (See Jones for the relationship of M-quantiles to quantiles
    in the case of non-redescending Norms.)

    Expectiles are M-quantiles with the LeastSquares as base norm.

    References
    ----------

    .. [*] Bianchi, Annamaria, and Nicola Salvati. 2015. “Asymptotic Properties
       and Variance Estimators of the M-Quantile Regression Coefficients
       Estimators.” Communications in Statistics - Theory and Methods 44 (11):
       2416–29. doi:10.1080/03610926.2013.791375.

    .. [*] Breckling, Jens, and Ray Chambers. 1988. “M-Quantiles.”
       Biometrika 75 (4): 761–71. doi:10.2307/2336317.

    .. [*] Jones, M. C. 1994. “Expectiles and M-Quantiles Are Quantiles.”
       Statistics & Probability Letters 20 (2): 149–53.
       doi:10.1016/0167-7152(94)90031-0.

    .. [*] Newey, Whitney K., and James L. Powell. 1987. “Asymmetric Least
       Squares Estimation and Testing.” Econometrica 55 (4): 819–47.
       doi:10.2307/1911031.
    """

    def __init__(self, q, base_norm):
        self.q = q
        self.base_norm = base_norm

    def _get_q(self, z):
        nobs = len(z)
        mask_neg = z < 0
        qq = np.empty(nobs)
        qq[mask_neg] = 1 - self.q
        qq[~mask_neg] = self.q
        return qq

    def rho(self, z):
        """
        The robust criterion function for MQuantileNorm.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
        """
        qq = self._get_q(z)
        return qq * self.base_norm.rho(z)

    def psi(self, z):
        """
        The psi function for MQuantileNorm estimator.

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
        """
        qq = self._get_q(z)
        return qq * self.base_norm.psi(z)

    def weights(self, z):
        """
        MQuantileNorm weighting function for the IRLS algorithm

        The psi function scaled by z, psi(z) / z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
        """
        qq = self._get_q(z)
        return qq * self.base_norm.weights(z)

    def psi_deriv(self, z):
        """
        The derivative of MQuantileNorm function

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi_deriv : ndarray

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        qq = self._get_q(z)
        return qq * self.base_norm.psi_deriv(z)

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)