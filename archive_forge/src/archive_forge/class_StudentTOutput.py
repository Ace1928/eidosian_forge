from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """
    args_dim: Dict[str, int] = {'df': 1, 'loc': 1, 'scale': 1}
    distribution_class: type = StudentT

    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        df = 2.0 + cls.squareplus(df)
        return (df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1))