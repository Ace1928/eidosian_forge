import gc
from typing import Tuple
import torch
Find a tensor from the heap

    Args:
        target_shape (tuple):
            Tensor shape to locate.
        only_param (bool):
            Only match Parameter type (e.g. for weights).

    Returns:
        (bool):
            Return True if found.
    