import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _reduce_scatter_tensor_coalesced(inputs: List[torch.Tensor], reduce_op: str, tag: str, ranks: List[int], group_size: int):
    group = c10d._find_or_create_pg_by_ranks_and_tag(tag, ranks, group_size)
    assert group is not None
    op = _str_to_reduce_op(reduce_op)

    def mk_out_tensor(shard):
        out_size = list(shard.size())
        out_size[0] //= group_size
        out_tensor = shard.new_empty(out_size)
        assert out_tensor.is_contiguous()
        return out_tensor
    out_tensors = [mk_out_tensor(t) for t in inputs]
    work_list = _reduce_scatter_tensor_coalesced_fallback(output_tensors=out_tensors, input_tensors=inputs, op=op, group=group, async_op=False)
    _register_tensor_work(out_tensors, work_list)
    return out_tensors