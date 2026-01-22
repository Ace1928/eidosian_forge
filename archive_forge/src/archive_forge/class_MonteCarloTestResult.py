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
@dataclass
class MonteCarloTestResult:
    """Result object returned by `scipy.stats.monte_carlo_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the sample.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray