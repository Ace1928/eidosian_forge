import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
class ProcessCovariance:
    """
    A covariance model for a process indexed by a real parameter.

    An implementation of this class is based on a positive definite
    correlation function h that maps real numbers to the interval [0,
    1], such as the Gaussian (squared exponential) correlation
    function :math:`\\exp(-x^2)`.  It also depends on a positive
    scaling function `s` and a positive smoothness function `u`.
    """

    def get_cov(self, time, sc, sm):
        """
        Returns the covariance matrix for given time values.

        Parameters
        ----------
        time : array_like
            The time points for the observations.  If len(time) = p,
            a pxp covariance matrix is returned.
        sc : array_like
            The scaling parameters for the observations.
        sm : array_like
            The smoothness parameters for the observation.  See class
            docstring for details.
        """
        raise NotImplementedError

    def jac(self, time, sc, sm):
        """
        The Jacobian of the covariance with respect to the parameters.

        See get_cov for parameters.

        Returns
        -------
        jsc : list-like
            jsc[i] is the derivative of the covariance matrix
            with respect to the i^th scaling parameter.
        jsm : list-like
            jsm[i] is the derivative of the covariance matrix
            with respect to the i^th smoothness parameter.
        """
        raise NotImplementedError