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
def _hypersphere_surface_sample(self, center: np.ndarray, radius: DecimalNumber, candidates: IntNumber=1) -> np.ndarray:
    """Uniform sampling on the hypersphere's surface."""
    vec = self.rng.standard_normal(size=(candidates, self.d))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    p = center + np.multiply(vec, radius)
    return p