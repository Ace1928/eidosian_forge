import torch
import torch.distributed as dist
from torch import nn
def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [torch.zeros_like(all_gather_in_list, device=all_gather_in_list.device, dtype=all_gather_in_list.dtype) for _ in range(world_size)]
    return out_list