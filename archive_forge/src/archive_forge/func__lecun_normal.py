import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def _lecun_normal(tensor, gain=1.0):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    denom = fan_in
    variance = gain / denom
    _no_grad_trunc_normal_(tensor, mean=0.0, std=math.sqrt(variance) / 0.8796256610342398, a=-2.0, b=2.0)