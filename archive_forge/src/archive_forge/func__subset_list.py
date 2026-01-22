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
def _subset_list(subset, whole_list, filter_items=None, warn=True, errors='raise'):
    """Handle list subsetting (var_names, groups...) across arviz.

    Parameters
    ----------
    subset : str, list, or None
    whole_list : list
        List from which to select a subset according to subset elements and
        filter_items value.
    filter_items : {None, "like", "regex"}, optional
        If `None` (default), interpret `subset` as the exact elements in `whole_list`
        names. If "like", interpret `subset` as substrings of the elements in
        `whole_list`. If "regex", interpret `subset` as regular expressions to match
        elements in `whole_list`. A la `pandas.filter`.
    errors: {"raise", "ignore"}, optional, default="raise"
        Select either to raise or ignore the invalid names.

    Returns
    -------
    list or None
        A subset of ``whole_list`` fulfilling the requests imposed by ``subset``
        and ``filter_items``.
    """
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        whole_list_tilde = [item for item in whole_list if _check_tilde_start(item)]
        if whole_list_tilde and warn:
            warnings.warn("ArviZ treats '~' as a negation character for selection. There are elements in `whole_list` starting with '~', {0}. Please double checkyour results to ensure all elements are included".format(', '.join(whole_list_tilde)))
        excluded_items = [item[1:] for item in subset if _check_tilde_start(item) and item not in whole_list]
        filter_items = str(filter_items).lower()
        if excluded_items:
            not_found = []
            if filter_items in {'like', 'regex'}:
                for pattern in excluded_items[:]:
                    excluded_items.remove(pattern)
                    if filter_items == 'like':
                        real_items = [real_item for real_item in whole_list if pattern in real_item]
                    else:
                        real_items = [real_item for real_item in whole_list if re.search(pattern, real_item)]
                    if not real_items:
                        not_found.append(pattern)
                    excluded_items.extend(real_items)
            not_found.extend([item for item in excluded_items if item not in whole_list])
            if not_found:
                warnings.warn(f'Items starting with ~: {not_found} have not been found and will be ignored')
            subset = [item for item in whole_list if item not in excluded_items]
        elif filter_items == 'like':
            subset = [item for item in whole_list for name in subset if name in item]
        elif filter_items == 'regex':
            subset = [item for item in whole_list for name in subset if re.search(name, item)]
        existing_items = np.isin(subset, whole_list)
        if not np.all(existing_items) and errors == 'raise':
            raise KeyError(f'{np.array(subset)[~existing_items]} are not present')
    return subset