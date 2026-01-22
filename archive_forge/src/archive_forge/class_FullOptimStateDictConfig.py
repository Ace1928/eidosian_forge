from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
@dataclass
class FullOptimStateDictConfig(OptimStateDictConfig):
    """
    Attributes:
        rank0_only (bool): If ``True``, then only rank 0 saves the full state
            dict, and nonzero ranks save an empty dict. If ``False``, then all
            ranks save the full state dict. (Default: ``False``)
    """
    rank0_only: bool = False