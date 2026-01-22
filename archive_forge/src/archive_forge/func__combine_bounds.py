import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _combine_bounds(name, user_bounds, shape_domain, integral):
    """Intersection of user-defined bounds and distribution PDF/PMF domain"""
    user_bounds = np.atleast_1d(user_bounds)
    if user_bounds[0] > user_bounds[1]:
        message = f'There are no values for `{name}` on the interval {list(user_bounds)}.'
        raise ValueError(message)
    bounds = (max(user_bounds[0], shape_domain[0]), min(user_bounds[1], shape_domain[1]))
    if integral and np.ceil(bounds[0]) > np.floor(bounds[1]):
        message = f'There are no integer values for `{name}` on the interval defined by the user-provided bounds and the domain of the distribution.'
        raise ValueError(message)
    elif not integral and bounds[0] > bounds[1]:
        message = f'There are no values for `{name}` on the interval defined by the user-provided bounds and the domain of the distribution.'
        raise ValueError(message)
    if not np.all(np.isfinite(bounds)):
        message = f'The intersection of user-provided bounds for `{name}` and the domain of the distribution is not finite. Please provide finite bounds for shape `{name}` in `bounds`.'
        raise ValueError(message)
    return bounds