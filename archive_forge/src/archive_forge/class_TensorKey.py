import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey(Key):
    """Hashable identifier for a storage which has been asigned an ID.

    A detailed description of Tensor IDs and why they are needed is given in
    `torch/csrc/profiler/collection.h` when `TensorID` is declared. To
    summarize, multiple Storage buffers can map to the same logical Tensor.
    This dataclass is used to refer to a concrete in-memory StorageImpl of
    a Tensor.
    """
    id: int
    storage: _Storage

    def __repr__(self) -> str:
        return f'id={self.id}: {repr(self.storage):<24} ({self.device})'

    def __lt__(self, other: 'TensorKey') -> bool:
        return self._as_sortable < other._as_sortable

    @staticmethod
    def _make(tensor_id: Optional[int], storage_ptr: Optional[int], allocation_id: Optional[int], device: torch.device) -> Optional['TensorKey']:
        if tensor_id is not None and storage_ptr is not None and (allocation_id is not None):
            return TensorKey(device, tensor_id, _Storage(storage_ptr, allocation_id))
        return None

    @classmethod
    def from_allocation(cls, alloc: _ExtraFields_Allocation) -> Optional['TensorKey']:
        return cls._make(alloc.id, alloc.ptr, alloc.allocation_id, alloc.device)

    @classmethod
    def from_tensor(cls, t: Optional[_TensorMetadata]) -> Optional['TensorKey']:
        if t is not None:
            return cls._make(t.id, t.storage_data_ptr, t.allocation_id, t.device)
        return None

    @property
    def _as_sortable(self) -> Tuple[int, int, str, int]:
        return (self.id, self.storage.allocation_id, self.device.type, self.device.index)