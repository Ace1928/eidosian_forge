from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from fairscale.internal import torch_version
from fairscale.nn.checkpoint import is_checkpointing, is_recomputing
def _track_running_stats(running_mean: Tensor, running_var: Tensor, momentum: float, mean: Tensor, var: Tensor, total_count: Tensor) -> None:
    unbiased_var = var * (total_count / (total_count - 1))
    running_mean += momentum * (mean.reshape(-1) - running_mean)
    running_var += momentum * (unbiased_var.reshape(-1) - running_var)