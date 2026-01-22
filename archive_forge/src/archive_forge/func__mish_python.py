import math
import warnings
from collections import OrderedDict
import torch
from packaging import version
from torch import Tensor, nn
from .utils import logging
def _mish_python(self, input: Tensor) -> Tensor:
    return input * torch.tanh(nn.functional.softplus(input))