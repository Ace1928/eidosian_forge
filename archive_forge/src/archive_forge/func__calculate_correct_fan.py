import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f'Mode {mode} not supported, please use one of {valid_modes}')
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out