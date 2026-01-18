import torch
from typing import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DTensor
def traverse_state_dict(state_dict: STATE_DICT_TYPE, visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None], keep_traversing: Callable[[STATE_DICT_ITEM], bool]=_keep_visiting_tensors) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.

    Traversal is short-circuited when if finds a collection for which ``keep_visiting_tensors`` evaluates
    to false for all elements.
    By default, all collections with at least one ``torch.Tensor`` element are traversed.
    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """

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

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)
    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)