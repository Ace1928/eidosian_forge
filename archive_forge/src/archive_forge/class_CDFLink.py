import numpy as np
import scipy.stats
import warnings
class CDFLink(Logit):
    """
    The use the CDF of a scipy.stats distribution

    CDFLink is a subclass of logit in order to use its _clean method
    for the link and its derivative.

    Parameters
    ----------
    dbn : scipy.stats distribution
        Default is dbn=scipy.stats.norm

    Notes
    -----
    The CDF link is untested.
    """

    def __init__(self, dbn=scipy.stats.norm):
        self.dbn = dbn

    def __call__(self, p):
        """
        CDF link function

        Parameters
        ----------
        p : array_like
            Mean parameters

        Returns
        -------
        z : ndarray
            (ppf) inverse of CDF transform of p

        Notes
        -----
        g(`p`) = `dbn`.ppf(`p`)
        """
        p = self._clean(p)
        return self.dbn.ppf(p)

    def inverse(self, z):
        """
        The inverse of the CDF link

        Parameters
        ----------
        z : array_like
            The value of the inverse of the link function at `p`

        Returns
        -------
        p : ndarray
            Mean probabilities.  The value of the inverse of CDF link of `z`

        Notes
        -----
        g^(-1)(`z`) = `dbn`.cdf(`z`)
        """
        return self.dbn.cdf(z)

    def deriv(self, p):
        """
        Derivative of CDF link

        Parameters
        ----------
        p : array_like
            mean parameters

        Returns
        -------
        g'(p) : ndarray
            The derivative of CDF transform at `p`

        Notes
        -----
        g'(`p`) = 1./ `dbn`.pdf(`dbn`.ppf(`p`))
        """
        p = self._clean(p)
        return 1.0 / self.dbn.pdf(self.dbn.ppf(p))

    def deriv2(self, p):
        """
        Second derivative of the link function g''(p)

        implemented through numerical differentiation
        """
        p = self._clean(p)
        linpred = self.dbn.ppf(p)
        return -self.inverse_deriv2(linpred) / self.dbn.pdf(linpred) ** 3

    def deriv2_numdiff(self, p):
        """
        Second derivative of the link function g''(p)

        implemented through numerical differentiation
        """
        from statsmodels.tools.numdiff import _approx_fprime_scalar
        p = np.atleast_1d(p)
        return _approx_fprime_scalar(p, self.deriv, centered=True)

    def inverse_deriv(self, z):
        """
        Derivative of the inverse link function

        Parameters
        ----------
        z : ndarray
            The inverse of the link function at `p`

        Returns
        -------
        g^(-1)'(z) : ndarray
            The value of the derivative of the inverse of the logit function.
            This is just the pdf in a CDFLink,
        """
        return self.dbn.pdf(z)

    def inverse_deriv2(self, z):
        """
        Second derivative of the inverse link function g^(-1)(z).

        Parameters
        ----------
        z : array_like
            `z` is usually the linear predictor for a GLM or GEE model.

        Returns
        -------
        g^(-1)''(z) : ndarray
            The value of the second derivative of the inverse of the link
            function

        Notes
        -----
        This method should be overwritten by subclasses.

        The inherited method is implemented through numerical differentiation.
        """
        from statsmodels.tools.numdiff import _approx_fprime_scalar
        z = np.atleast_1d(z)
        return _approx_fprime_scalar(z, self.inverse_deriv, centered=True)