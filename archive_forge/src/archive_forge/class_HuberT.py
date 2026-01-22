import numpy as np
class HuberT(RobustNorm):
    """
    Huber's T for M estimation.

    Parameters
    ----------
    t : float, optional
        The tuning constant for Huber's t function. The default value is
        1.345.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, t=1.345):
        self.t = t

    def _subset(self, z):
        """
        Huber's T is defined piecewise over the range for z
        """
        z = np.asarray(z)
        return np.less_equal(np.abs(z), self.t)

    def rho(self, z):
        """
        The robust criterion function for Huber's t.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = .5*z**2            for \\|z\\| <= t

            rho(z) = \\|z\\|*t - .5*t**2    for \\|z\\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * 0.5 * z ** 2 + (1 - test) * (np.abs(z) * self.t - 0.5 * self.t ** 2)

    def psi(self, z):
        """
        The psi function for Huber's t estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z      for \\|z\\| <= t

            psi(z) = sign(z)*t for \\|z\\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z + (1 - test) * self.t * np.sign(z)

    def weights(self, z):
        """
        Huber's t weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1          for \\|z\\| <= t

            weights(z) = t/\\|z\\|      for \\|z\\| > t
        """
        z_isscalar = np.isscalar(z)
        z = np.atleast_1d(z)
        test = self._subset(z)
        absz = np.abs(z)
        absz[test] = 1.0
        v = test + (1 - test) * self.t / absz
        if z_isscalar:
            v = v[0]
        return v

    def psi_deriv(self, z):
        """
        The derivative of Huber's t psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.less_equal(np.abs(z), self.t).astype(float)