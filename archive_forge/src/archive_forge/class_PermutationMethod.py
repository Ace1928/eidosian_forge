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
class PermutationMethod(ResamplingMethod):
    """Configuration information for a permutation hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a permutation version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, then that instance is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.
    """
    random_state: object = None

    def _asdict(self):
        return dict(n_resamples=self.n_resamples, batch=self.batch, random_state=self.random_state)