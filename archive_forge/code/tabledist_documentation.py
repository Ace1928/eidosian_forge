import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf
from statsmodels.tools.decorators import cache_readonly

        Returns interpolated quantiles, similar to ppf or isf

        uses Rbf to interpolate critical values as function of `prob` and `n`

        Parameters
        ----------
        prob : array_like
            probabilities corresponding to the definition of table columns
        n : int or float
            sample size, second parameter of the table

        Returns
        -------
        ppf : array_like
            critical values with same shape as prob, returns nan for arguments
            that are outside of the table bounds
        