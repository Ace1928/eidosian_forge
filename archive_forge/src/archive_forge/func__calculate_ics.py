import itertools
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Mapping, cast, Callable
import numpy as np
import pandas as pd
import scipy.stats as st
from xarray_einstats import stats
import xarray as xr
from scipy.optimize import minimize
from typing_extensions import Literal
from .. import _log
from ..data import InferenceData, convert_to_dataset, convert_to_inference_data, extract
from ..rcparams import rcParams, ScaleKeyword, ICKeyword
from ..utils import Numba, _numba_var, _var_names, get_coords
from .density_utils import get_bins as _get_bins
from .density_utils import histogram as _histogram
from .density_utils import kde as _kde
from .diagnostics import _mc_error, _multichain_statistics, ess
from .stats_utils import ELPDData, _circular_standard_deviation, smooth_data
from .stats_utils import get_log_likelihood as _get_log_likelihood
from .stats_utils import get_log_prior as _get_log_prior
from .stats_utils import logsumexp as _logsumexp
from .stats_utils import make_ufunc as _make_ufunc
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
from ..sel_utils import xarray_var_iter
from ..labels import BaseLabeller
def _calculate_ics(compare_dict, scale: Optional[ScaleKeyword]=None, ic: Optional[ICKeyword]=None, var_name: Optional[str]=None):
    """Calculate LOO or WAIC only if necessary.

    It always calls the ic function with ``pointwise=True``.

    Parameters
    ----------
    compare_dict :  dict of {str : InferenceData or ELPDData}
        A dictionary of model names and InferenceData or ELPDData objects
    scale : str, optional
        Output scale for IC. Available options are:

        - `log` : (default) log-score (after Vehtari et al. (2017))
        - `negative_log` : -1 * (log-score)
        - `deviance` : -2 * (log-score)

        A higher log-score (or a lower deviance) indicates a model with better predictive accuracy.
    ic : str, optional
        Information Criterion (PSIS-LOO `loo` or WAIC `waic`) used to compare models.
        Defaults to ``rcParams["stats.information_criterion"]``.
    var_name : str, optional
        Name of the variable storing pointwise log likelihood values in ``log_likelihood`` group.


    Returns
    -------
    compare_dict : dict of ELPDData
    scale : str
    ic : str

    """
    precomputed_elpds = {name: elpd_data for name, elpd_data in compare_dict.items() if isinstance(elpd_data, ELPDData)}
    precomputed_ic = None
    precomputed_scale = None
    if precomputed_elpds:
        _, arbitrary_elpd = precomputed_elpds.popitem()
        precomputed_ic = arbitrary_elpd.index[0].split('_')[1]
        precomputed_scale = arbitrary_elpd['scale']
        raise_non_pointwise = f'{precomputed_ic}_i' not in arbitrary_elpd
        if any((elpd_data.index[0].split('_')[1] != precomputed_ic for elpd_data in precomputed_elpds.values())):
            raise ValueError('All information criteria to be compared must be the same but found both loo and waic.')
        if any((elpd_data['scale'] != precomputed_scale for elpd_data in precomputed_elpds.values())):
            raise ValueError('All information criteria to be compared must use the same scale')
        if any((f'{precomputed_ic}_i' not in elpd_data for elpd_data in precomputed_elpds.values())) or raise_non_pointwise:
            raise ValueError('Not all provided ELPDData have been calculated with pointwise=True')
        if ic is not None and ic.lower() != precomputed_ic:
            warnings.warn(f'Provided ic argument is incompatible with precomputed elpd data. Using ic from precomputed elpddata: {precomputed_ic}')
            ic = precomputed_ic
        if scale is not None and scale.lower() != precomputed_scale:
            warnings.warn(f'Provided scale argument is incompatible with precomputed elpd data. Using scale from precomputed elpddata: {precomputed_scale}')
            scale = precomputed_scale
    if ic is None and precomputed_ic is None:
        ic = cast(ICKeyword, rcParams['stats.information_criterion'])
    elif ic is None:
        ic = precomputed_ic
    else:
        ic = cast(ICKeyword, ic.lower())
    allowable = ['loo', 'waic'] if NO_GET_ARGS else get_args(ICKeyword)
    if ic not in allowable:
        raise ValueError(f'{ic} is not a valid value for ic: must be in {allowable}')
    if scale is None and precomputed_scale is None:
        scale = cast(ScaleKeyword, rcParams['stats.ic_scale'])
    elif scale is None:
        scale = precomputed_scale
    else:
        scale = cast(ScaleKeyword, scale.lower())
    allowable = ['log', 'negative_log', 'deviance'] if NO_GET_ARGS else get_args(ScaleKeyword)
    if scale not in allowable:
        raise ValueError(f'{scale} is not a valid value for scale: must be in {allowable}')
    if ic == 'loo':
        ic_func: Callable = loo
    elif ic == 'waic':
        ic_func = waic
    else:
        raise NotImplementedError(f'The information criterion {ic} is not supported.')
    compare_dict = deepcopy(compare_dict)
    for name, dataset in compare_dict.items():
        if not isinstance(dataset, ELPDData):
            try:
                compare_dict[name] = ic_func(convert_to_inference_data(dataset), pointwise=True, scale=scale, var_name=var_name)
            except Exception as e:
                raise e.__class__(f'Encountered error trying to compute {ic} from model {name}.') from e
    return (compare_dict, scale, ic)