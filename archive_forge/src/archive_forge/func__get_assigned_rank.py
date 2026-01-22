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
def _get_assigned_rank(self, bucket_index: int) -> int:
    """
        Return the single rank assigned to a :class:`DistributedDataParallel` gradient bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                bucket for which to get the assigned rank.
        """
    assert not self._overlap_info.shard_buckets, 'The bucket assignment requires global bucket information and will be computed later; there should be no need to use this method'
    return bucket_index % self.world_size