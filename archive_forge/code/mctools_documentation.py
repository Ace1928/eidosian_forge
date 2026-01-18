from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable
summary table for cumulative density function


        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            probabilities for which
        crit : array_like
            values for which cdf is calculated
        varnames : None, or list of strings
            optional list of variable names, same length as idx

        Returns
        -------
        table : instance of SimpleTable
            use `print(table` to see results


        