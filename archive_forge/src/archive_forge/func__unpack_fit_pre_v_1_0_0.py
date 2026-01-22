import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def _unpack_fit_pre_v_1_0_0(fit, items, save_warmup, dtypes):
    """Transform fit to dictionary containing ndarrays.

    Parameters
    ----------
    data: cmdstanpy.CmdStanMCMC
    items: list
    save_warmup: bool
    dtypes: dict

    Returns
    -------
    dict
        key, values pairs. Values are formatted to shape = (chains, draws, *shape)
    """
    num_warmup = 0
    if save_warmup:
        if not fit._save_warmup:
            save_warmup = False
        else:
            num_warmup = fit.num_draws_warmup
    nchains = fit.chains
    draws = np.swapaxes(fit.draws(inc_warmup=save_warmup), 0, 1)
    sample = {}
    sample_warmup = {}
    stan_vars_cols = fit.metadata.stan_vars_cols if hasattr(fit, 'metadata') else fit.stan_vars_cols
    sampler_vars_cols = fit.metadata._method_vars_cols if hasattr(fit, 'metadata') else fit.sampler_vars_cols
    for item in items:
        if item in stan_vars_cols:
            col_idxs = stan_vars_cols[item]
            raw_draws = fit.stan_variable(item, inc_warmup=save_warmup)
            raw_draws = np.swapaxes(raw_draws.reshape((-1, nchains, *raw_draws.shape[1:]), order='F'), 0, 1)
        elif item in sampler_vars_cols:
            col_idxs = sampler_vars_cols[item]
            raw_draws = draws[..., col_idxs[0]]
        else:
            raise ValueError(f'fit data, unknown variable: {item}')
        raw_draws = raw_draws.astype(dtypes.get(item))
        if save_warmup:
            sample_warmup[item] = raw_draws[:, :num_warmup, ...]
            sample[item] = raw_draws[:, num_warmup:, ...]
        else:
            sample[item] = raw_draws
    return (sample, sample_warmup)