from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
@property
def _sst_enabled(self) -> bool:
    """True if SST is enabled."""
    return self._sst_top_k_element is not None or self._sst_top_k_percent is not None