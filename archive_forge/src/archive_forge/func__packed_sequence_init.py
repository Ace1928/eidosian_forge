import warnings
from typing import Iterable, List, NamedTuple, Tuple, Union
import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional
def _packed_sequence_init(data: Tensor, batch_sizes: Optional[Tensor]=None, sorted_indices: Optional[Tensor]=None, unsorted_indices: Optional[Tensor]=None) -> PackedSequence:
    data, batch_sizes, sorted_indices, unsorted_indices = _packed_sequence_init_args(data, batch_sizes, sorted_indices, unsorted_indices)
    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)