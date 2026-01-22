from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def _gather_ipc_meta(self, shard_data):
    all_data = [None] * self.world_size
    dist.all_gather_object(all_data, shard_data)
    handles = []
    offsets = []
    for i in range(len(all_data)):
        handles.append(all_data[i][0])
        offsets.append(all_data[i][1])
    return (handles, offsets)