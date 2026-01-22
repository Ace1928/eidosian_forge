from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """
    args_dim: Dict[str, int] = {'loc': 1, 'scale': 1}
    distribution_class: type = Normal

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        return (loc.squeeze(-1), scale.squeeze(-1))