import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Union, overload
import torch
import torch.nn as nn
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
def _gen_param_group_key(param_keys: List[str]) -> str:
    """Concatenate all param keys as a unique indentifier for one param group."""
    return '/'.join(sorted(param_keys))