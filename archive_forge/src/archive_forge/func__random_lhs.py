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
def _random_lhs(self, n: IntNumber=1) -> np.ndarray:
    """Base LHS algorithm."""
    if not self.scramble:
        samples: np.ndarray | float = 0.5
    else:
        samples = self.rng.uniform(size=(n, self.d))
    perms = np.tile(np.arange(1, n + 1), (self.d, 1))
    for i in range(self.d):
        self.rng.shuffle(perms[i, :])
    perms = perms.T
    samples = (perms - samples) / n
    return samples