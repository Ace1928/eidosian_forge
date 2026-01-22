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
def _build_param_buckets(self) -> None:
    """
        Build parameter buckets if ``parameters_as_bucket_view=True``.

        For each device that stores this rank's parameters, there is a
        bucket (represented as a tensor) containing all of the parameters on
        that device that are assigned to a given rank in the parameter update
        partition.

        This method is called in the constructor and any time parameter
        trainability is changed.

        .. warning::
            The current implementation assumes that all of the parameters in a
            bucket are of the same dense type when allocating the bucket's
            tensor.

        .. warning::
            If the model parameters are stored across more than one device,
            then the storage partitioning must be the same across all
            processes in order for parameter synchronization to work.
        """
    if not self.parameters_as_bucket_view or self._overlap_with_ddp:
        return
    num_devices = len(self._device_to_params_per_rank)
    self._buckets = [[] for _ in range(num_devices)]
    for dev_i, (device, params_per_rank) in enumerate(self._device_to_params_per_rank.items()):
        for params in params_per_rank:
            bucket_size = 0
            dtype = None
            trainable_params = []
            for param in params:
                if not _is_trainable(param):
                    param.data = param.data.detach().clone()
                else:
                    bucket_size += param.numel()
                    trainable_params.append(param)
                dtype = param.dtype
            if bucket_size == 0:
                bucket = torch.zeros(1, device=device)
            else:
                bucket = torch.empty(bucket_size, dtype=dtype, device=device)
                offset = 0
                for param in trainable_params:
                    offset_next = offset + param.numel()
                    bucket[offset:offset_next].copy_(param.data.flatten())
                    param.data = bucket[offset:offset_next].view_as(param.data)
                    offset = offset_next
            self._buckets[dev_i].append(bucket)