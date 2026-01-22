import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _all_gather_into_tensor_coalesced(self, tag, rankset, group_size):
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, rankset, group_size)
    assert group is not None

    def mk_out_tensor(shard):
        out_size = list(shard.size())
        out_size[0] *= group_size
        out_tensor = shard.new_empty(out_size)
        assert out_tensor.is_contiguous()
        return out_tensor
    out_tensors = [mk_out_tensor(t) for t in self]
    work_list = _all_gather_into_tensor_coalesced_fallback(output_tensors=out_tensors, input_tensors=self, group=group, async_op=True)
    _register_tensor_work(out_tensors, work_list)
    return out_tensors