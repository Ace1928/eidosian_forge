import numpy as np
import scipy.stats
import warnings
class Logit(Link):
    """
    The logit transform

    Notes
    -----
    call and derivative use a private method _clean to make trim p by
    machine epsilon so that p is in (0,1)

    Alias of Logit:
    logit = Logit()
    """

    def _clean(self, p):
        """
        Clip logistic values to range (eps, 1-eps)

        Parameters
        ----------
        p : array_like
            Probabilities

        Returns
        -------
        pclip : ndarray
            Clipped probabilities
        """
        return np.clip(p, FLOAT_EPS, 1.0 - FLOAT_EPS)

    def __call__(self, p):
        """
        The logit transform

        Parameters
        ----------
        p : array_like
            Probabilities

        Returns
        -------
        z : ndarray
            Logit transform of `p`

        Notes
        -----
        g(p) = log(p / (1 - p))
        """
        p = self._clean(p)
        return np.log(p / (1.0 - p))

    def inverse(self, z):
        """
        Inverse of the logit transform

        Parameters
        ----------
        z : array_like
            The value of the logit transform at `p`

        Returns
        -------
        p : ndarray
            Probabilities

        Notes
        -----
        g^(-1)(z) = exp(z)/(1+exp(z))
        """
        z = np.asarray(z)
        t = np.exp(-z)
        return 1.0 / (1.0 + t)

    def deriv(self, p):
        """
        Derivative of the logit transform

        Parameters
        ----------
        p : array_like
            Probabilities

        Returns
        -------
        g'(p) : ndarray
            Value of the derivative of logit transform at `p`

        Notes
        -----
        g'(p) = 1 / (p * (1 - p))

        Alias for `Logit`:
        logit = Logit()
        """
        p = self._clean(p)
        return 1.0 / (p * (1 - p))

    def inverse_deriv(self, z):
        """
        Derivative of the inverse of the logit transform

        Parameters
        ----------
        z : array_like
            `z` is usually the linear predictor for a GLM or GEE model.

        Returns
        -------
        g'^(-1)(z) : ndarray
            The value of the derivative of the inverse of the logit function
        """
        t = np.exp(z)
        return t / (1 + t) ** 2

    def deriv2(self, p):
        """
        Second derivative of the logit function.

        Parameters
        ----------
        p : array_like
            probabilities

        Returns
        -------
        g''(z) : ndarray
            The value of the second derivative of the logit function
        """
        v = p * (1 - p)
        return (2 * p - 1) / v ** 2