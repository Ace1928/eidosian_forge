import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
def _build_ddp_param_buckets(self) -> None:
    """
        Build the DDP bucket with parameters assigned to this rank.

        For each DDP bucket with parameters assigned to this rank, flattens the
        data of those parameters into a single tensor and saves the tensor to
        the ``tensor`` attribute in the corresponding
        :class:`_DDPBucketAssignment` instance stored in
        ``self._bucket_assignments_per_rank``.

        :class:`DistributedDataParallel` guarantees that the parameters
        corresponding to a gradient bucket have the same device and the same
        dtype.
        """
    for bucket_assignments in self._bucket_assignments_per_rank:
        for bucket_assignment in bucket_assignments.values():
            params = bucket_assignment.parameters
            bucket_size = 0
            dtype = None
            for param in params:
                assert _is_trainable(param), 'Model parameter corresponding to a gradient in a DDP bucket should require a gradient'
                bucket_size += param.numel()
                dtype = param.dtype
            assert bucket_size > 0, 'Empty bucket'
            tensor = torch.empty(bucket_size, dtype=dtype, device=bucket_assignment.device)
            offset = 0
            for param in params:
                offset_next = offset + param.numel()
                tensor[offset:offset_next].copy_(param.data.flatten())
                param.data = tensor[offset:offset_next].view_as(param.data)
                offset = offset_next
            bucket_assignment.tensor = tensor