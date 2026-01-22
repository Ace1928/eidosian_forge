import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _free_storage(tensor: torch.Tensor) -> None:
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        already_freed = tensor._typed_storage()._size() == 0
        if not already_freed:
            _p_assert(tensor.storage_offset() == 0, f"Freeing a tensor's storage is unsafe when it is not the sole occupant\nstorage offset: {tensor.storage_offset()}\nstorage size: {tensor._typed_storage()._size()}\ntensor shape: {tensor.shape}")
            tensor._typed_storage()._resize_(0)