import torch
from typing import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DTensor
def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
    if _is_terminal(value):
        visitor(path, value)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _traverse_obj(path + (str(k),), v)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _traverse_obj(path + (i,), v)