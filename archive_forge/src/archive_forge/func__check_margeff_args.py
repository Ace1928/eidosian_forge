from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _check_margeff_args(at, method):
    """
    Checks valid options for margeff
    """
    if at not in ['overall', 'mean', 'median', 'zero', 'all']:
        raise ValueError('%s not a valid option for `at`.' % at)
    if method not in ['dydx', 'eyex', 'dyex', 'eydx']:
        raise ValueError('method is not understood.  Got %s' % method)