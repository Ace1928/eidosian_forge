import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_1_13, _XLA_AVAILABLE
from torchmetrics.utilities.prints import rank_zero_warn
def _cumsum(x: Tensor, dim: Optional[int]=0, dtype: Optional[torch.dtype]=None) -> Tensor:
    if torch.are_deterministic_algorithms_enabled() and x.is_cuda and x.is_floating_point() and (sys.platform != 'win32'):
        rank_zero_warn('You are trying to use a metric in deterministic mode on GPU that uses `torch.cumsum`, which is currently not supported. The tensor will be copied to the CPU memory to compute it and then copied back to GPU. Expect some slowdowns.', TorchMetricsUserWarning)
        return x.cpu().cumsum(dim=dim, dtype=dtype).cuda()
    return torch.cumsum(x, dim=dim, dtype=dtype)