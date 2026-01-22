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
def _multichain_statistics(ary, focus='mean'):
    """Calculate efficiently multichain statistics for summary.

    Parameters
    ----------
    ary : numpy.ndarray
    focus : select focus for the statistics. Deafault is mean.

    Returns
    -------
    tuple
        Order of return parameters is
            If focus equals "mean"
                - mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat
            Else if focus equals "median"
                - mcse_median, ess_median, ess_tail, r_hat
    """
    ary = np.atleast_2d(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        if focus == 'mean':
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        return (np.nan, np.nan, np.nan, np.nan)
    z_split = _z_scale(_split_chains(ary))
    quantile05, quantile95 = _quantile(ary, [0.05, 0.95])
    iquantile05 = ary <= quantile05
    quantile05_ess = _ess(_split_chains(iquantile05))
    iquantile95 = ary <= quantile95
    quantile95_ess = _ess(_split_chains(iquantile95))
    ess_tail_value = min(quantile05_ess, quantile95_ess)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=2)):
        rhat_value = np.nan
    else:
        rhat_bulk = _rhat(z_split)
        ary_folded = np.abs(ary - np.median(ary))
        rhat_tail = _rhat(_z_scale(_split_chains(ary_folded)))
        rhat_value = max(rhat_bulk, rhat_tail)
    if focus == 'mean':
        ess_mean_value = _ess_mean(ary)
        ess_sd_value = _ess_sd(ary)
        sd = np.std(ary, ddof=1)
        mcse_mean_value = sd / np.sqrt(ess_mean_value)
        ess_bulk_value = _ess(z_split)
        fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / ess_sd_value) ** (ess_sd_value - 1) - 1)
        mcse_sd_value = sd * fac_mcse_sd
        return (mcse_mean_value, mcse_sd_value, ess_bulk_value, ess_tail_value, rhat_value)
    ess_median_value = _ess_median(ary)
    mcse_median_value = _mcse_median(ary)
    return (mcse_median_value, ess_median_value, ess_tail_value, rhat_value)