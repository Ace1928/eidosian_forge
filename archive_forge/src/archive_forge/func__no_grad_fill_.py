import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def _no_grad_fill_(tensor, val):
    with torch.no_grad():
        return tensor.fill_(val)