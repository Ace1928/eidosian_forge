import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        already_allocated = tensor._typed_storage()._size() == size.numel()
        if not already_allocated:
            tensor_storage_size = tensor._typed_storage()._size()
            _p_assert(tensor_storage_size == 0, f'Tensor storage should have been resized to be 0 but got {tensor_storage_size}')
            tensor._typed_storage()._resize_(size.numel())