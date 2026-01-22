from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def all_reduce_unreg(self, inp: torch.Tensor, out: torch.Tensor=None):
    if out is None:
        out = torch.empty_like(inp)
    custom_ar.all_reduce_unreg(self._ptr, inp, self.buffer, out)
    return out