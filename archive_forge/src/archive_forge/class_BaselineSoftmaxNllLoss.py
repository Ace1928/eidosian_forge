from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class BaselineSoftmaxNllLoss(BaselineSoftmax):
    """Baseline that does an output projection, a softmax & a NLL loss (cross-entropy).

    See BaselineSoftmax above. Constructor is the same. Only difference is in the
    forward function.

    This class is used for testing and benchmarking.
    """

    def __init__(self, proj_weight: nn.Parameter, tile_factor: int=0, log_softmax: bool=True, margin: float=0.35, scale: Optional[float]=None):
        super().__init__(proj_weight, tile_factor, log_softmax, margin, scale)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward that directly compute the loss."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        x = super().forward(input, target)
        return F.nll_loss(x, target, reduction='sum')