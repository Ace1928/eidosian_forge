import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _register_tensor_wrapper(tensor) -> None:
    global data_ptr_to_work
    data_ptr = tensor.elem.data_ptr()
    wait_reg = data_ptr_to_work.get(data_ptr, None)
    if wait_reg is None:
        warnings.warn('Trying to register finalizer to AsyncCollectiveTensor but the inner tensor is already gone')
    else:
        wait_reg._record_wrapper(data_ptr)
        weakref.finalize(tensor, _wait_reg_dec, data_ptr, wait_reg)