import numpy as np
from numpy.linalg import svd
import scipy
import pandas as pd
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from .multivariate_ols import multivariate_stats
class CanCorrTestResults:
    """
    Canonical correlation results class

    Attributes
    ----------
    stats : DataFrame
        Contain statistical tests results for each canonical correlation
    stats_mv : DataFrame
        Contain the multivariate statistical tests results
    """

    def __init__(self, stats, stats_mv):
        self.stats = stats
        self.stats_mv = stats_mv

    def __str__(self):
        return self.summary().__str__()

    def summary(self):
        summ = summary2.Summary()
        summ.add_title('Cancorr results')
        summ.add_df(self.stats)
        summ.add_dict({'': ''})
        summ.add_dict({'Multivariate Statistics and F Approximations': ''})
        summ.add_df(self.stats_mv)
        return summ