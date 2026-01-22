import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def _make_tensor(self, params, state):
    size, steps, allocation_size = self._get_size_and_steps(params)
    constructor = self._tensor_constructor or self.default_tensor_constructor
    raw_tensor = constructor(size=allocation_size, dtype=self._dtype, **params)
    if self._cuda:
        raw_tensor = raw_tensor.cuda()
    dim = len(size)
    order = np.arange(dim)
    if state.rand() > self._probability_contiguous:
        while dim > 1 and np.all(order == np.arange(dim)):
            order = state.permutation(raw_tensor.dim())
        raw_tensor = raw_tensor.permute(tuple(order)).contiguous()
        raw_tensor = raw_tensor.permute(tuple(np.argsort(order)))
    slices = [slice(0, size * step, step) for size, step in zip(size, steps)]
    tensor = raw_tensor[slices]
    properties = {'numel': int(tensor.numel()), 'order': order, 'steps': steps, 'is_contiguous': tensor.is_contiguous(), 'dtype': str(self._dtype)}
    return (tensor, properties)