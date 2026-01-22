from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
def all_gather_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool=False):
    world_size = torch.distributed.get_world_size(process_group)
    output = torch.empty(world_size * input_.shape[0], *input_.shape[1:], dtype=input_.dtype, device=input_.device)
    handle = torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=process_group, async_op=async_op)
    return (output, handle)