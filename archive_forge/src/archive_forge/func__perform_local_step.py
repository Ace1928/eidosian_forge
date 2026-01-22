import weakref
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import (
from torch.nn.parallel.distributed import DistributedDataParallel
def _perform_local_step(bucket: dist.GradBucket, zero: ZeroRedundancyOptimizer, rank: int):
    """
    Performs a local optimizer step using the gradients provided by ``bucket``.

    Arguments:
        bucket (dist.GradBucket): the bucket providing the gradients.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to perform the :meth:`_local_step`.
        rank (int): the calling process's rank.

    .. warning::
        This function assumes that appropriate synchronization has taken place
        so that the bucket's gradients can be used.
    """
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    assert len(zero.optim.param_groups) == 1, 'Overlapping DDP with ZeRO only supports a single parameter group'
    num_local_optim_params = len(zero.optim.param_groups[0]['params'])
    gradients: List[Optional[torch.Tensor]] = [_NO_PARAM_UPDATE for _ in range(num_local_optim_params)]
    assert bucket_index in overlap_info.offsets, f'Bucket index {bucket_index} was not assigned to rank {rank}'
    gradients_offset = overlap_info.offsets[bucket_index]
    bucket_assignment = zero._bucket_assignments_per_rank[rank][bucket_index]
    bucket_offset = bucket_assignment.offset
    length = len(bucket_assignment.parameters)
    bucket_gradients = bucket.gradients()[bucket_offset:bucket_offset + length]
    for i, grad in enumerate(bucket_gradients):
        gradients[gradients_offset + i] = grad
    zero._local_step(gradients)