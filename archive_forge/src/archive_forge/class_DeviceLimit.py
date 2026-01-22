import math
from dataclasses import dataclass, field
from typing import Mapping, Tuple
import torch
@dataclass
class DeviceLimit:
    name: str = 'default'
    source: str = ''
    sm: Tuple[int, int] = (0, 0)
    gmem_bandwidth: float = math.inf
    gemm_tflops: Mapping[torch.dtype, float] = field(default_factory=dict)