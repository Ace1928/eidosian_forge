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
def _calculate_null_samples(data, statistic, n_permutations, batch, random_state=None):
    """
    Calculate null distribution for paired-sample tests.
    """
    n_samples = len(data)
    if n_samples == 1:
        data = [data[0], -data[0]]
    data = np.swapaxes(data, 0, -1)

    def statistic_wrapped(*data, axis):
        data = np.swapaxes(data, 0, -1)
        if n_samples == 1:
            data = data[0:1]
        return statistic(*data, axis=axis)
    return _calculate_null_pairings(data, statistic_wrapped, n_permutations, batch, random_state)