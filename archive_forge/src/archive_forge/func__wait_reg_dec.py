import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _wait_reg_dec(ptr, wait_reg):
    wait_reg.decrement_live_tensor(ptr)