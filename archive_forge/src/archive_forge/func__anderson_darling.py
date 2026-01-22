import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _anderson_darling(dist, data):
    x = np.sort(data, axis=-1)
    n = data.shape[-1]
    i = np.arange(1, n + 1)
    Si = (2 * i - 1) / n * (dist.logcdf(x) + dist.logsf(x[..., ::-1]))
    S = np.sum(Si, axis=-1)
    return -n - S