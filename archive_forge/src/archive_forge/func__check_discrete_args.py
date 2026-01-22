from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _check_discrete_args(at, method):
    """
    Checks the arguments for margeff if the exogenous variables are discrete.
    """
    if method in ['dyex', 'eyex']:
        raise ValueError('%s not allowed for discrete variables' % method)
    if at in ['median', 'zero']:
        raise ValueError('%s not allowed for discrete variables' % at)