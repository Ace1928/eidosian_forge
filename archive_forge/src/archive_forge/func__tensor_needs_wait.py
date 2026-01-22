import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _tensor_needs_wait(tensor: torch.Tensor) -> bool:
    """Returns true if ```tensor``` needs to be waited. Works with ACS and inner tensors."""
    if hasattr(tensor, '_get_acs_underlying_tensor'):
        tensor = tensor._get_acs_underlying_tensor()
    data_ptr = tensor.data_ptr()
    wait_reg = data_ptr_to_work.get(data_ptr)
    return wait_reg is not None and wait_reg.work is not None