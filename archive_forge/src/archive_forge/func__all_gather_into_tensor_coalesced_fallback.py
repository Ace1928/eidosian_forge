import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _all_gather_into_tensor_coalesced_fallback(output_tensors, input_tensors, group, async_op=False):
    if input_tensors[0].is_cpu or not async_op:
        work_list = []
        out_tensors_sliced = [list(torch.chunk(out_tensor, dist.get_world_size(group))) for out_tensor in output_tensors]
        for shard, out_tensor in zip(input_tensors, out_tensors_sliced):
            work = c10d.all_gather(out_tensor, shard, group=group, async_op=async_op)
            work_list.append(work)
        return work_list
    else:
        with c10d._coalescing_manager(group=group, async_ops=True) as cm:
            for in_t, out_t in zip(input_tensors, output_tensors):
                dist.all_gather_into_tensor(out_t, in_t, group=group, async_op=True)
        return cm