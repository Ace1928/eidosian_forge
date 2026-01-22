import functools
from typing import Dict, Optional, Tuple, Union
import torch
from torch.cuda import _CudaDeviceProperties
@functools.lru_cache(None)
def _properties() -> Dict[int, _CudaDeviceProperties]:
    if not torch.cuda.is_available():
        return {}
    try:
        return {i: torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())}
    except RuntimeError:
        return {}