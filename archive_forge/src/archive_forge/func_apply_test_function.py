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
def apply_test_function(idata, func, group='both', var_names=None, pointwise=False, out_data_shape=None, out_pp_shape=None, out_name_data='T', out_name_pp=None, func_args=None, func_kwargs=None, ufunc_kwargs=None, wrap_data_kwargs=None, wrap_pp_kwargs=None, inplace=True, overwrite=None):
    """Apply a Bayesian test function to an InferenceData object.

    Parameters
    ----------
    idata: InferenceData
        :class:`arviz.InferenceData` object on which to apply the test function.
        This function will add new variables to the InferenceData object
        to store the result without modifying the existing ones.
    func: callable
        Callable that calculates the test function. It must have the following call signature
        ``func(y, theta, *args, **kwargs)`` (where ``y`` is the observed data or posterior
        predictive and ``theta`` the model parameters) even if not all the arguments are
        used.
    group: str, optional
        Group on which to apply the test function. Can be observed_data, posterior_predictive
        or both.
    var_names: dict group -> var_names, optional
        Mapping from group name to the variables to be passed to func. It can be a dict of
        strings or lists of strings. There is also the option of using ``both`` as key,
        in which case, the same variables are used in observed data and posterior predictive
        groups
    pointwise: bool, optional
        If True, apply the test function to each observation and sample, otherwise, apply
        test function to each sample.
    out_data_shape, out_pp_shape: tuple, optional
        Output shape of the test function applied to the observed/posterior predictive data.
        If None, the default depends on the value of pointwise.
    out_name_data, out_name_pp: str, optional
        Name of the variables to add to the observed_data and posterior_predictive datasets
        respectively. ``out_name_pp`` can be ``None``, in which case will be taken equal to
        ``out_name_data``.
    func_args: sequence, optional
        Passed as is to ``func``
    func_kwargs: mapping, optional
        Passed as is to ``func``
    wrap_data_kwargs, wrap_pp_kwargs: mapping, optional
        kwargs passed to :func:`~arviz.wrap_xarray_ufunc`. By default, some suitable input_core_dims
        are used.
    inplace: bool, optional
        If True, add the variables inplace, otherwise, return a copy of idata with the variables
        added.
    overwrite: bool, optional
        Overwrite data in case ``out_name_data`` or ``out_name_pp`` are already variables in
        dataset. If ``None`` it will be the opposite of inplace.

    Returns
    -------
    idata: InferenceData
        Output InferenceData object. If ``inplace=True``, it is the same input object modified
        inplace.

    See Also
    --------
    plot_bpv :  Plot Bayesian p-value for observed data and Posterior/Prior predictive.

    Notes
    -----
    This function is provided for convenience to wrap scalar or functions working on low
    dims to inference data object. It is not optimized to be faster nor as fast as vectorized
    computations.

    Examples
    --------
    Use ``apply_test_function`` to wrap ``numpy.min`` for illustration purposes. And plot the
    results.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> az.apply_test_function(idata, lambda y, theta: np.min(y))
        >>> T = idata.observed_data.T.item()
        >>> az.plot_posterior(idata, var_names=["T"], group="posterior_predictive", ref_val=T)

    """
    out = idata if inplace else deepcopy(idata)
    valid_groups = ('observed_data', 'posterior_predictive', 'both')
    if group not in valid_groups:
        raise ValueError(f'Invalid group argument. Must be one of {valid_groups} not {group}.')
    if overwrite is None:
        overwrite = not inplace
    if out_name_pp is None:
        out_name_pp = out_name_data
    if func_args is None:
        func_args = tuple()
    if func_kwargs is None:
        func_kwargs = {}
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault('check_shape', False)
    ufunc_kwargs.setdefault('ravel', False)
    if wrap_data_kwargs is None:
        wrap_data_kwargs = {}
    if wrap_pp_kwargs is None:
        wrap_pp_kwargs = {}
    if var_names is None:
        var_names = {}
    both_var_names = var_names.pop('both', None)
    var_names.setdefault('posterior', list(out.posterior.data_vars))
    in_posterior = out.posterior[var_names['posterior']]
    if isinstance(in_posterior, xr.Dataset):
        in_posterior = in_posterior.to_array().squeeze()
    groups = ('posterior_predictive', 'observed_data') if group == 'both' else [group]
    for grp in groups:
        out_group_shape = out_data_shape if grp == 'observed_data' else out_pp_shape
        out_name_group = out_name_data if grp == 'observed_data' else out_name_pp
        wrap_group_kwargs = wrap_data_kwargs if grp == 'observed_data' else wrap_pp_kwargs
        if not hasattr(out, grp):
            raise ValueError(f'InferenceData object must have {grp} group')
        if not overwrite and out_name_group in getattr(out, grp).data_vars:
            raise ValueError(f'Should overwrite: {out_name_group} variable present in group {grp}, but overwrite is False')
        var_names.setdefault(grp, list(getattr(out, grp).data_vars) if both_var_names is None else both_var_names)
        in_group = getattr(out, grp)[var_names[grp]]
        if isinstance(in_group, xr.Dataset):
            in_group = in_group.to_array(dim=f'{grp}_var').squeeze()
        if pointwise:
            out_group_shape = in_group.shape if out_group_shape is None else out_group_shape
        elif grp == 'observed_data':
            out_group_shape = () if out_group_shape is None else out_group_shape
        elif grp == 'posterior_predictive':
            out_group_shape = in_group.shape[:2] if out_group_shape is None else out_group_shape
        loop_dims = in_group.dims[:len(out_group_shape)]
        wrap_group_kwargs.setdefault('input_core_dims', [[dim for dim in dataset.dims if dim not in loop_dims] for dataset in [in_group, in_posterior]])
        func_kwargs['out'] = np.empty(out_group_shape)
        out_group = getattr(out, grp)
        try:
            out_group[out_name_group] = _wrap_xarray_ufunc(func, in_group.values, in_posterior.values, func_args=func_args, func_kwargs=func_kwargs, ufunc_kwargs=ufunc_kwargs, **wrap_group_kwargs)
        except IndexError:
            excluded_dims = set(wrap_group_kwargs['input_core_dims'][0] + wrap_group_kwargs['input_core_dims'][1])
            out_group[out_name_group] = _wrap_xarray_ufunc(func, *xr.broadcast(in_group, in_posterior, exclude=excluded_dims), func_args=func_args, func_kwargs=func_kwargs, ufunc_kwargs=ufunc_kwargs, **wrap_group_kwargs)
        setattr(out, grp, out_group)
    return out