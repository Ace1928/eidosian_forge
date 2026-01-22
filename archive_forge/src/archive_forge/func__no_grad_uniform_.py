import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def _no_grad_uniform_(tensor, a, b, generator=None):
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=generator)