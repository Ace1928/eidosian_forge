import torch
import numbers
from torch.nn.parameter import Parameter
from .module import Module
from ._functions import CrossMapLRN2d as _cross_map_lrn2d
from .. import functional as F
from .. import init
from torch import Tensor, Size
from typing import Union, List, Tuple
class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(self, size: int, alpha: float=0.0001, beta: float=0.75, k: float=1) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return _cross_map_lrn2d.apply(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self) -> str:
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)