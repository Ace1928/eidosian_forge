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
def _random_oa_lhs(self, n: IntNumber=4) -> np.ndarray:
    """Orthogonal array based LHS of strength 2."""
    p = np.sqrt(n).astype(int)
    n_row = p ** 2
    n_col = p + 1
    primes = primes_from_2_to(p + 1)
    if p not in primes or n != n_row:
        raise ValueError(f'n is not the square of a prime number. Close values are {primes[-2:] ** 2}')
    if self.d > p + 1:
        raise ValueError('n is too small for d. Must be n > (d-1)**2')
    oa_sample = np.zeros(shape=(n_row, n_col), dtype=int)
    arrays = np.tile(np.arange(p), (2, 1))
    oa_sample[:, :2] = np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, 2)
    for p_ in range(1, p):
        oa_sample[:, 2 + p_ - 1] = np.mod(oa_sample[:, 0] + p_ * oa_sample[:, 1], p)
    oa_sample_ = np.empty(shape=(n_row, n_col), dtype=int)
    for j in range(n_col):
        perms = self.rng.permutation(p)
        oa_sample_[:, j] = perms[oa_sample[:, j]]
    oa_lhs_sample = np.zeros(shape=(n_row, n_col))
    lhs_engine = LatinHypercube(d=1, scramble=self.scramble, strength=1, seed=self.rng)
    for j in range(n_col):
        for k in range(p):
            idx = oa_sample[:, j] == k
            lhs = lhs_engine.random(p).flatten()
            oa_lhs_sample[:, j][idx] = lhs + oa_sample[:, j][idx]
            lhs_engine = lhs_engine.reset()
    oa_lhs_sample /= p
    return oa_lhs_sample[:, :self.d]