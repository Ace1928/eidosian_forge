from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
def unpack_non_tensors(tensors: Tuple[torch.Tensor, ...], packed_non_tensors: Optional[Dict[str, List[Any]]]) -> Tuple[Any, ...]:
    """See split_non_tensors."""
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict), type(packed_non_tensors)
    mixed: List[Any] = []
    is_tensor_list = packed_non_tensors['is_tensor']
    objects = packed_non_tensors['objects']
    assert len(tensors) + len(objects) == len(is_tensor_list), f'len(tensors) {len(tensors)} len(objects) {len(objects)} len(is_tensor_list) {len(is_tensor_list)}'
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)