from __future__ import annotations
import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
import numpy as np
import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
from ._qmc_cy import (
def _ensure_in_unit_hypercube(sample: npt.ArrayLike) -> np.ndarray:
    """Ensure that sample is a 2D array and is within a unit hypercube

    Parameters
    ----------
    sample : array_like (n, d)
        A 2D array of points.

    Returns
    -------
    np.ndarray
        The array interpretation of the input sample

    Raises
    ------
    ValueError
        If the input is not a 2D array or contains points outside of
        a unit hypercube.
    """
    sample = np.asarray(sample, dtype=np.float64, order='C')
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')
    if sample.max() > 1.0 or sample.min() < 0.0:
        raise ValueError('Sample is not in unit hypercube')
    return sample