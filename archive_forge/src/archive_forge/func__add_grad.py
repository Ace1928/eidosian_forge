import torch
from functools import reduce
from .optimizer import Optimizer
def _add_grad(self, step_size, update):
    offset = 0
    for p in self._params:
        numel = p.numel()
        p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
        offset += numel
    assert offset == self._numel()