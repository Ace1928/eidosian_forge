import math
import warnings
from typing import Any, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from peft.tuners.lycoris_utils import LycorisLayer, check_adapters_to_merge
def _block_diagonal(self, oft_r: torch.Tensor, rank: int) -> torch.Tensor:
    if oft_r.shape[0] == 1:
        blocks = [oft_r[0, ...] for i in range(rank)]
    else:
        blocks = [oft_r[i, ...] for i in range(rank)]
    A = torch.block_diag(*blocks)
    return A