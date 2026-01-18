import numpy as np
import scipy.stats
import warnings

        Derivative of the inverse of the negative binomial transform

        Parameters
        ----------
        z : array_like
            Usually the linear predictor for a GLM or GEE model

        Returns
        -------
        g^(-1)'(z) : ndarray
            The value of the derivative of the inverse of the negative
            binomial link
        