import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def _get_size_and_steps(self, params):
    dim = params[self._dim_parameter] if self._dim_parameter is not None else len(self._size)

    def resolve(values, dim):
        """Resolve values into concrete integers."""
        values = tuple((params.get(i, i) for i in values))
        if len(values) > dim:
            values = values[:dim]
        if len(values) < dim:
            values = values + tuple((1 for _ in range(dim - len(values))))
        return values
    size = resolve(self._size, dim)
    steps = resolve(self._steps or (), dim)
    allocation_size = tuple((size_i * step_i for size_i, step_i in zip(size, steps)))
    return (size, steps, allocation_size)