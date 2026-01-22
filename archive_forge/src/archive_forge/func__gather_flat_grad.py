import torch
from functools import reduce
from .optimizer import Optimizer
def _gather_flat_grad(self):
    views = []
    for p in self._params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().view(-1)
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)