from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _check_at_is_all(method):
    if method['at'] == 'all':
        raise ValueError("Only margeff are available when `at` is 'all'. Please input specific points if you would like to do inference.")