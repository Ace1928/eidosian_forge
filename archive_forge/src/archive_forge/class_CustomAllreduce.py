from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
class CustomAllreduce:

    def __init__(self, rank, world_size, max_size=8192 * 1024) -> None:
        self.meta = torch.zeros(custom_ar.meta_size() + max_size, dtype=torch.uint8, device='cuda')
        self.buffer = torch.empty(max_size, dtype=torch.uint8, device='cuda')
        self.rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device='cuda')
        self.max_size = max_size
        self.world_size = world_size
        handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = _is_full_nvlink(rank, world_size)
        self._ptr = custom_ar.init_custom_ar(self.meta, self.rank_data, handles, offsets, rank, self.full_nvlink)
        self.fast_cond = self.full_nvlink or world_size <= 2
        self.register_buffer(self.buffer)

    def _get_ipc_meta(self, inp: torch.Tensor):
        data = inp.untyped_storage()._share_cuda_()
        shard_data = (data[1], data[3])
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        all_data = [None] * self.world_size
        dist.all_gather_object(all_data, shard_data)
        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0])
            offsets.append(all_data[i][1])
        return (handles, offsets)

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        custom_ar.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        handle, offset = custom_ar.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
        logger.info('Registering %d cuda graph addresses', len(offset))
        custom_ar.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        return custom_ar.should_custom_ar(inp, self.max_size, self.world_size, self.full_nvlink)

    def all_reduce_reg(self, inp: torch.Tensor, out: torch.Tensor=None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_reg(self._ptr, inp, out)
        return out

    def all_reduce_unreg(self, inp: torch.Tensor, out: torch.Tensor=None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def close(self):
        if self._ptr:
            custom_ar.dispose(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()