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
def _permutation_test_iv(data, statistic, permutation_type, vectorized, n_resamples, batch, alternative, axis, random_state):
    """Input validation for `permutation_test`."""
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    permutation_types = {'samples', 'pairings', 'independent'}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f'`permutation_type` must be in {permutation_types}.')
    if vectorized not in {True, False, None}:
        raise ValueError('`vectorized` must be `True`, `False`, or `None`.')
    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters
    if not vectorized:
        statistic = _vectorize_statistic(statistic)
    message = '`data` must be a tuple containing at least two samples'
    try:
        if len(data) < 2 and permutation_type == 'independent':
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)
    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError('each sample in `data` must contain two or more observations along `axis`.')
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)
    n_resamples_int = int(n_resamples) if not np.isinf(n_resamples) else np.inf
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
    random_state = check_random_state(random_state)
    return (data_iv, statistic, permutation_type, vectorized, n_resamples_int, batch_iv, alternative, axis_int, random_state)