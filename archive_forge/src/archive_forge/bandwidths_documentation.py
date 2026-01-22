import numpy as np
from scipy.stats import scoreatpercentile
from statsmodels.compat.pandas import Substitution
from statsmodels.sandbox.nonparametric import kernels

    Selects bandwidth for a selection rule bw

    this is a wrapper around existing bandwidth selection rules

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    bw : str
        name of bandwidth selection rule, currently supported are:
        %s
    kernel : not used yet

    Returns
    -------
    bw : float
        The estimate of the bandwidth
    