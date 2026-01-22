import torch
from typing import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DTensor
def _is_terminal(value: STATE_DICT_ITEM) -> bool:
    values: Collection[STATE_DICT_ITEM]
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, list):
        values = value
    else:
        return True
    for entry in values:
        if isinstance(entry, (Mapping, list)) and (not _is_terminal(entry)):
            return False
        if keep_traversing is not None and keep_traversing(entry):
            return False
    return True