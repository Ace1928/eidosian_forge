import warnings
from .distance import PairwiseDistance
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction
from torch import Tensor
from typing import Callable, Optional
class _WeightedLoss(_Loss):

    def __init__(self, weight: Optional[Tensor]=None, size_average=None, reduce=None, reduction: str='mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]