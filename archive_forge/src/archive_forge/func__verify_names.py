import warnings
from collections import OrderedDict
import numpy as np
import xarray as xr
from .. import utils
from .base import dict_to_dataset, generate_dims_coords, make_attrs
from .inference_data import InferenceData
def _verify_names(sampler, var_names, arg_names, slices):
    """Make sure var_names and arg_names are assigned reasonably.

    This is meant to run before loading emcee objects into InferenceData.
    In case var_names or arg_names is None, will provide defaults. If they are
    not None, it verifies there are the right number of them.

    Throws a ValueError in case validation fails.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Fitted emcee sampler
    var_names : list[str] or None
        Names for the emcee parameters
    arg_names : list[str] or None
        Names for the args/observations provided to emcee
    slices : list[seq] or None
        slices to select the variables (used for multidimensional variables)

    Returns
    -------
    list[str], list[str], list[seq]
        Defaults for var_names, arg_names and slices
    """
    if hasattr(sampler, 'args'):
        ndim = sampler.chain.shape[-1]
        num_args = len(sampler.args)
    elif hasattr(sampler, 'log_prob_fn'):
        ndim = sampler.get_chain().shape[-1]
        num_args = len(sampler.log_prob_fn.args)
    else:
        ndim = sampler.get_chain().shape[-1]
        num_args = 0
    if slices is None:
        slices = utils.arange(ndim)
        num_vars = ndim
    else:
        num_vars = len(slices)
    indices = utils.arange(ndim)
    slicing_try = np.concatenate([utils.one_de(indices[idx]) for idx in slices])
    if len(set(slicing_try)) != ndim:
        warnings.warn(f'Check slices: Not all parameters in chain captured. {ndim} are present, and {len(slicing_try)} have been captured.', UserWarning)
    if len(slicing_try) != len(set(slicing_try)):
        warnings.warn(f'Overlapping slices. Check the index present: {slicing_try}', UserWarning)
    if var_names is None:
        var_names = [f'var_{idx}' for idx in range(num_vars)]
    if arg_names is None:
        arg_names = [f'arg_{idx}' for idx in range(num_args)]
    if len(var_names) != num_vars:
        raise ValueError(f'The sampler has {num_vars} variables, but only {len(var_names)} var_names were provided!')
    if len(arg_names) != num_args:
        raise ValueError(f'The sampler has {num_args} args, but only {len(arg_names)} arg_names were provided!')
    return (var_names, arg_names, slices)