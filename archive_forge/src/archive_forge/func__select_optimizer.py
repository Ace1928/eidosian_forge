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
def _select_optimizer(optimization: Literal['random-cd', 'lloyd'] | None, config: dict) -> Callable | None:
    """A factory for optimization methods."""
    optimization_method: dict[str, Callable] = {'random-cd': _random_cd, 'lloyd': _lloyd_centroidal_voronoi_tessellation}
    optimizer: partial | None
    if optimization is not None:
        try:
            optimization = optimization.lower()
            optimizer_ = optimization_method[optimization]
        except KeyError as exc:
            message = f'{optimization!r} is not a valid optimization method. It must be one of {set(optimization_method)!r}'
            raise ValueError(message) from exc
        optimizer = partial(optimizer_, **config)
    else:
        optimizer = None
    return optimizer