import copy
import itertools
import math
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
def _get_remote_device_str(rank, device_type, num_devices_per_node):
    if device_type.lower() == 'cpu':
        return f'rank:{rank}/{device_type}'
    else:
        return f'rank:{rank}/{device_type}:{rank % num_devices_per_node}'