import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning

        Profile-likelihood inference for variance parameters.

        Parameters
        ----------
        re_ix : int
            If vtype is `re`, this value is the index of the variance
            parameter for which to construct a profile likelihood.  If
            `vtype` is 'vc' then `re_ix` is the name of the variance
            parameter to be profiled.
        vtype : str
            Either 're' or 'vc', depending on whether the profile
            analysis is for a random effect or a variance component.
        num_low : int
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : int
            The number of points at which to calculate the likelihood
            above the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        An array with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.

        Notes
        -----
        Only variance parameters can be profiled.
        