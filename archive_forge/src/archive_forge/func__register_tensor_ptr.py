import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _register_tensor_ptr(self, data_ptr):
    global data_ptr_to_work
    data_ptr_to_work[data_ptr] = self
    self.ptrs.add(data_ptr)