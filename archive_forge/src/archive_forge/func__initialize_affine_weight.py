from typing import Callable, Optional
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .initialize import get_model_parallel_rank, get_model_parallel_world_size
from .mappings import (
from .utils import VocabUtility, divide_and_check_no_remainder
def _initialize_affine_weight(weight: torch.Tensor, out_features: int, in_features: int, per_partition_size: int, partition_dim: int, init_method: Callable[[torch.Tensor], torch.Tensor], stride: int=1, return_master_weight: bool=False) -> Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)
    per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None