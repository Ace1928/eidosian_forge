from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _get_string_subset(data, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Get a subset from a string array.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return names that match this regular expression.

    Returns
    -------
    data : list-like, shape<=[n_features]
        List of matching strings
    """
    data = utils.toarray(data)
    mask = _get_string_subset_mask(data, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex)
    return data[mask]