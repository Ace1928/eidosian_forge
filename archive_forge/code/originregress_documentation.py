import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.tools.tools import add_constant

        Returns the confidence interval for a regression parameter when the
        regression is forced through the origin.

        Parameters
        ----------
        param_num : int
            The parameter number to be tested.  Note this uses python
            indexing but the '0' parameter refers to the intercept term.
        upper_bound : float
            The maximum value the upper confidence limit can be.  The
            closer this is to the confidence limit, the quicker the
            computation.  Default is .00001 confidence limit under normality.
        lower_bound : float
            The minimum value the lower confidence limit can be.
            Default is .00001 confidence limit under normality.
        sig : float, optional
            The significance level.  Default .05.
        method : str, optional
             Algorithm to optimize of nuisance params.  Can be 'nm' or
            'powell'.  Default is 'nm'.
        stochastic_exog : bool
            Default is True.

        Returns
        -------
        ci: tuple
            The confidence interval for the parameter 'param_num'.
        