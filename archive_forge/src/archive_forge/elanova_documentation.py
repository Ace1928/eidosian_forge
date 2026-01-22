import numpy as np
from .descriptive import _OptFuncts
from scipy import optimize
from scipy.stats import chi2

        Returns -2 log likelihood, the pvalue and the maximum likelihood
        estimate for a common mean.

        Parameters
        ----------

        mu : float
            If a mu is specified, ANOVA is conducted with mu as the
            common mean.  Otherwise, the common mean is the maximum
            empirical likelihood estimate of the common mean.
            Default is None.

        mu_start : float
            Starting value for commean mean if specific mu is not specified.
            Default = 0.

        return_weights : bool
            if TRUE, returns the weights on observations that maximize the
            likelihood.  Default is FALSE.

        Returns
        -------

        res: tuple
            The log-likelihood, p-value and estimate for the common mean.
        