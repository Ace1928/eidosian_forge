from dataclasses import dataclass
from typing import List
import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed.remote_device import _remote_device

        Creates a Shard of a ShardedTensor from a local torch.Tensor, shard_offsets and rank.

        Args:
            tensor(torch.Tensor): Local tensor for the shard.
            shard_offsets(List[int]): List of integers specify the offset
                of the shard on each dimension.
            rank(int): Specify the rank for the shard.
        