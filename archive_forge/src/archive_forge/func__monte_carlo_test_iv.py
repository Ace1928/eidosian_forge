from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
def _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples, batch, alternative, axis):
    """Input validation for `monte_carlo_test`."""
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    if vectorized not in {True, False, None}:
        raise ValueError('`vectorized` must be `True`, `False`, or `None`.')
    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        data = (data,)
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError('`rvs` must be callable or sequence of callables.')
    if not len(rvs) == len(data):
        message = 'If `rvs` is a sequence, `len(rvs)` must equal `len(data)`.'
        raise ValueError(message)
    if not callable(statistic):
        raise TypeError('`statistic` must be callable.')
    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters
    if not vectorized:
        statistic_vectorized = _vectorize_statistic(statistic)
    else:
        statistic_vectorized = statistic
    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)
    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError('`n_resamples` must be a positive integer.')
    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError('`batch` must be a positive integer or None.')
    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be in {alternatives}')
    return (data_iv, rvs, statistic_vectorized, vectorized, n_resamples_int, batch_iv, alternative, axis_int)