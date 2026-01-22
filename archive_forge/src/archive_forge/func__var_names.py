import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
def _var_names(var_names, data, filter_vars=None, errors='raise'):
    """Handle var_names input across arviz.

    Parameters
    ----------
    var_names: str, list, or None
    data : xarray.Dataset
        Posterior data in an xarray
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
         interpret var_names as substrings of the real variables names. If "regex",
         interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    errors: {"raise", "ignore"}, optional, default="raise"
        Select either to raise or ignore the invalid names.

    Returns
    -------
    var_name: list or None
    """
    if filter_vars not in {None, 'like', 'regex'}:
        raise ValueError(f"'filter_vars' can only be None, 'like', or 'regex', got: '{filter_vars}'")
    if errors not in {'raise', 'ignore'}:
        raise ValueError(f"'errors' can only be 'raise', or 'ignore', got: '{errors}'")
    if var_names is not None:
        if isinstance(data, (list, tuple)):
            all_vars = []
            for dataset in data:
                dataset_vars = list(dataset.data_vars)
                for var in dataset_vars:
                    if var not in all_vars:
                        all_vars.append(var)
        else:
            all_vars = list(data.data_vars)
        all_vars_tilde = [var for var in all_vars if _check_tilde_start(var)]
        if all_vars_tilde:
            warnings.warn("ArviZ treats '~' as a negation character for variable selection.\n                   Your model has variables names starting with '~', {0}. Please double check\n                   your results to ensure all variables are included".format(', '.join(all_vars_tilde)))
        try:
            var_names = _subset_list(var_names, all_vars, filter_items=filter_vars, warn=False, errors=errors)
        except KeyError as err:
            msg = ' '.join(('var names:', f'{err}', 'in dataset'))
            raise KeyError(msg) from err
    return var_names