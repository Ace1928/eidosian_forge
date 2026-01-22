import math
import warnings
from numbers import Number
from typing import Optional, Union
import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.distributions.utils import lazy_property
def _mvdigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    assert x.gt((p - 1) / 2).all(), 'Wrong domain for multivariate digamma function.'
    return torch.digamma(x.unsqueeze(-1) - torch.arange(p, dtype=x.dtype, device=x.device).div(2).expand(x.shape + (-1,))).sum(-1)