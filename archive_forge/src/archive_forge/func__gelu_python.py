import math
import warnings
from collections import OrderedDict
import torch
from packaging import version
from torch import Tensor, nn
from .utils import logging
def _gelu_python(self, input: Tensor) -> Tensor:
    return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))