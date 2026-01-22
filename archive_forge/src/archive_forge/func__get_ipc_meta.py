from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def _get_ipc_meta(self, inp: torch.Tensor):
    data = inp.untyped_storage()._share_cuda_()
    shard_data = (data[1], data[3])
    return self._gather_ipc_meta(shard_data)