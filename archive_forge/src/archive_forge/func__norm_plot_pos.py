import warnings
import numpy as np
import pandas as pd
from scipy import stats
def _norm_plot_pos(observations):
    """
    Computes standard normal (Gaussian) plotting positions using scipy.

    Parameters
    ----------
    observations : array_like
        Sequence of observed quantities.

    Returns
    -------
    plotting_position : array of floats
    """
    ppos, sorted_res = stats.probplot(observations, fit=False)
    return stats.norm.cdf(ppos)