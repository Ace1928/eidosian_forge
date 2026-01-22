import numpy as np
class RobustNorm:
    """
    The parent class for the norms used for robust regression.

    Lays out the methods expected of the robust norms to be used
    by statsmodels.RLM.

    See Also
    --------
    statsmodels.rlm

    Notes
    -----
    Currently only M-estimators are available.

    References
    ----------
    PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York, 1981.

    DC Montgomery, EA Peck. 'Introduction to Linear Regression Analysis',
        John Wiley and Sons, Inc., New York, 2001.

    R Venables, B Ripley. 'Modern Applied Statistics in S'
        Springer, New York, 2002.
    """

    def rho(self, z):
        """
        The robust criterion estimator function.

        Abstract method:

        -2 loglike used in M-estimator
        """
        raise NotImplementedError

    def psi(self, z):
        """
        Derivative of rho.  Sometimes referred to as the influence function.

        Abstract method:

        psi = rho'
        """
        raise NotImplementedError

    def weights(self, z):
        """
        Returns the value of psi(z) / z

        Abstract method:

        psi(z) / z
        """
        raise NotImplementedError

    def psi_deriv(self, z):
        """
        Derivative of psi.  Used to obtain robust covariance matrix.

        See statsmodels.rlm for more information.

        Abstract method:

        psi_derive = psi'
        """
        raise NotImplementedError

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)