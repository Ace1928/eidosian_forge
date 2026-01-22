import math
from dataclasses import dataclass
from typing import (
import torch
def _create_block_mask(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
    return _materialize_causal_mask(shape, dtype=dtype, device=device, window_size=self._window_size, from_bottomright=True)