from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
def _reshape_inputs(input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D inputs to 2D for this kernel"""
    if len(input.shape) == 3:
        input = input.reshape(-1, input.shape[2])
    if len(target.shape) == 2:
        target = target.reshape(-1)
    return (input, target)