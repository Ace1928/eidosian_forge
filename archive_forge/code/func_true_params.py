from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.genmod.generalized_estimating_equations import GEE,\
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import GlobalOddsRatio
from .gee_gaussian_simulation_check import GEE_simulator
def true_params(self):
    return np.concatenate(self.params[:-1])