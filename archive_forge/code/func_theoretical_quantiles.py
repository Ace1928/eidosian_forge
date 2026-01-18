from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
@cache_readonly
def theoretical_quantiles(self):
    """Theoretical quantiles"""
    try:
        return self.dist.ppf(self.theoretical_percentiles)
    except TypeError:
        msg = f'{self.dist.name} requires more parameters to compute ppf'
        raise TypeError(msg)
    except Exception as exc:
        msg = f'failed to compute the ppf of {self.dist.name}'
        raise type(exc)(msg)