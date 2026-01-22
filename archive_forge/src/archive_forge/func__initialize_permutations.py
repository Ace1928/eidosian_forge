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
def _initialize_permutations(self) -> None:
    """Initialize permutations for all Van der Corput sequences.

        Permutations are only needed for scrambling.
        """
    self._permutations: list = [None] * len(self.base)
    if self.scramble:
        for i, bdim in enumerate(self.base):
            permutations = _van_der_corput_permutations(base=bdim, random_state=self.rng)
            self._permutations[i] = permutations