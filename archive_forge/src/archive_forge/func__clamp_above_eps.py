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
def _clamp_above_eps(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=torch.finfo(x.dtype).eps)