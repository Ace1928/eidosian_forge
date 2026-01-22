from typing import TypeVar, cast
import torch
from torch import Tensor, nn
from torch.nn.functional import batch_norm
from torch.nn.modules.batchnorm import _BatchNorm
from .checkpoint import is_recomputing
def _track(self, input: Tensor) -> bool:
    """Tracks statistics of a micro-batch."""
    dim = [0]
    dim.extend(range(2, input.dim()))
    with torch.no_grad():
        self.sum += input.sum(dim)
        self.sum_squares += (input ** 2).sum(dim)
    size = input.size().numel() // input.size(1)
    self.counter += size
    self.tracked += 1
    return self.tracked == self.chunks