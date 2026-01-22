from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def _get_filter_idx(values, cutoff, percentile, keep_cells):
    """Return a boolean array to index cells based on a filter.

    Parameters
    ----------
    values : list-like, shape=[n_samples]
        Value upon which to filter
    cutoff : float or tuple of floats, optional (default: None)
        Value above or below which to retain cells. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain cells.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above, below or between the cutoff.
        If None, defaults to 'above' when a single cutoff is given and
        'between' when two cutoffs are given.

    Returns
    -------
    keep_cells_idx : list-like
        Boolean retention array
    """
    cutoff = _get_percentile_cutoff(values, cutoff, percentile, required=True)
    if keep_cells is None:
        if isinstance(cutoff, numbers.Number):
            keep_cells = 'above'
        else:
            keep_cells = 'between'
    if keep_cells == 'above':
        if not isinstance(cutoff, numbers.Number):
            raise ValueError("Expected a single cutoff with keep_cells='above'. Got {}".format(cutoff))
        keep_cells_idx = values > cutoff
    elif keep_cells == 'below':
        if not isinstance(cutoff, numbers.Number):
            raise ValueError("Expected a single cutoff with keep_cells='below'. Got {}".format(cutoff))
        keep_cells_idx = values < cutoff
    elif keep_cells == 'between':
        if isinstance(cutoff, numbers.Number) or len(cutoff) != 2:
            raise ValueError("Expected cutoff of length 2 with keep_cells='between'. Got {}".format(cutoff))
        keep_cells_idx = np.logical_and(values > np.min(cutoff), values < np.max(cutoff))
    else:
        raise ValueError("Expected `keep_cells` in ['above', 'below', 'between']. Got {}".format(keep_cells))
    return keep_cells_idx