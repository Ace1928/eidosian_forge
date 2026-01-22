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
def _z_scale(ary):
    """Calculate z_scale.

    Parameters
    ----------
    ary : np.ndarray

    Returns
    -------
    np.ndarray
    """
    ary = np.asarray(ary)
    if packaging.version.parse(scipy.__version__) < packaging.version.parse('1.10.0.dev0'):
        rank = stats.rankdata(ary, method='average')
    else:
        rank = stats.rankdata(ary, method='average', nan_policy='omit')
    rank = _backtransform_ranks(rank)
    z = stats.norm.ppf(rank)
    z = z.reshape(ary.shape)
    return z