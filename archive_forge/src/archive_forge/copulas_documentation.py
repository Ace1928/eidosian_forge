from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from statsmodels.graphics import utils
Compute correlation parameter from tau.

        Parameters
        ----------
        tau : float
            Kendall's tau.

        Returns
        -------
        corr_param : float
            Correlation parameter of the copula, ``theta`` in Archimedean and
            pearson correlation in elliptical.

        