import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
def _zeros_float32(self, element_count: int, use_numpy: bool) -> NumpyArrayOrPyTorchTensor:
    if use_numpy:
        import numpy as np
        return np.zeros(element_count, dtype=np.float32)
    else:
        return torch.zeros(element_count, dtype=torch.float32, device='cpu')