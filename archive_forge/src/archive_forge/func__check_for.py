from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
def _check_for(dist, attr='ppf'):
    if not hasattr(dist, attr):
        raise AttributeError(f'distribution must have a {attr} method')