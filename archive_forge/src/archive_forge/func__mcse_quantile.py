import warnings
from collections.abc import Sequence
import numpy as np
import packaging
import pandas as pd
import scipy
from scipy import stats
from ..data import convert_to_dataset
from ..utils import Numba, _numba_var, _stack, _var_names
from .density_utils import histogram as _histogram
from .stats_utils import _circular_standard_deviation, _sqrt
from .stats_utils import autocov as _autocov
from .stats_utils import not_valid as _not_valid
from .stats_utils import quantile as _quantile
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
def _mcse_quantile(ary, prob):
    """Compute the Markov Chain quantile error at quantile=prob."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    ess = _ess_quantile(ary, prob)
    probability = [0.1586553, 0.8413447]
    with np.errstate(invalid='ignore'):
        ppf = stats.beta.ppf(probability, ess * prob + 1, ess * (1 - prob) + 1)
    sorted_ary = np.sort(ary.ravel())
    size = sorted_ary.size
    ppf_size = ppf * size - 1
    th1 = sorted_ary[int(np.floor(np.nanmax((ppf_size[0], 0))))]
    th2 = sorted_ary[int(np.ceil(np.nanmin((ppf_size[1], size - 1))))]
    return (th2 - th1) / 2