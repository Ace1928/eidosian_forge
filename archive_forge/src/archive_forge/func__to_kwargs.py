import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _to_kwargs(inputs: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]], target_device: torch.device, use_side_stream_for_tensor_copies: bool) -> Tuple[Tuple[Any, ...], Tuple[Dict[str, Any], ...]]:
    moved_inputs = _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies) if inputs else []
    moved_kwargs = _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies) if kwargs else []
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    elif len(moved_kwargs) < len(moved_inputs):
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    return (tuple(moved_inputs), tuple(moved_kwargs))