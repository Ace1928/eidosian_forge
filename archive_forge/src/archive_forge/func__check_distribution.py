import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def _check_distribution(self, distribution):
    if not isinstance(distribution, dict):
        assert distribution in _DISTRIBUTIONS
    else:
        assert not any((i < 0 for i in distribution.values())), 'Probabilities cannot be negative'
        assert abs(sum(distribution.values()) - 1) <= 1e-05, 'Distribution is not normalized'
        assert self._minval is None
        assert self._maxval is None
    return distribution