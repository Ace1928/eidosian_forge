import numpy as np
class TukeyBiweight(RobustNorm):
    """

    Tukey's biweight function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Tukey's Biweight.  The default value is
        c = 4.685.

    Notes
    -----
    Tukey's biweight is sometime's called bisquare.
    """

    def __init__(self, c=4.685):
        self.c = c

    def _subset(self, z):
        """
        Tukey's biweight is defined piecewise over the range of z
        """
        z = np.abs(np.asarray(z))
        return np.less_equal(z, self.c)

    def rho(self, z):
        """
        The robust criterion function for Tukey's biweight estimator

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = -(1 - (z/c)**2)**3 * c**2/6.   for \\|z\\| <= R

            rho(z) = 0                              for \\|z\\| > R
        """
        subset = self._subset(z)
        factor = self.c ** 2 / 6.0
        return -(1 - (z / self.c) ** 2) ** 3 * subset * factor + factor

    def psi(self, z):
        """
        The psi function for Tukey's biweight estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z*(1 - (z/c)**2)**2        for \\|z\\| <= R

            psi(z) = 0                           for \\|z\\| > R
        """
        z = np.asarray(z)
        subset = self._subset(z)
        return z * (1 - (z / self.c) ** 2) ** 2 * subset

    def weights(self, z):
        """
        Tukey's biweight weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            psi(z) = (1 - (z/c)**2)**2          for \\|z\\| <= R

            psi(z) = 0                          for \\|z\\| > R
        """
        subset = self._subset(z)
        return (1 - (z / self.c) ** 2) ** 2 * subset

    def psi_deriv(self, z):
        """
        The derivative of Tukey's biweight psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        subset = self._subset(z)
        return subset * ((1 - (z / self.c) ** 2) ** 2 - 4 * z ** 2 / self.c ** 2 * (1 - (z / self.c) ** 2))