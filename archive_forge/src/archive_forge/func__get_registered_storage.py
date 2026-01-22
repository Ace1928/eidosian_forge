from __future__ import annotations
import dataclasses
from . import torch_wrapper
def _get_registered_storage(self, pointer: torch.Tensor):
    max_pointer = torch.max(pointer).item()
    min_pointer = torch.min(pointer).item()
    registered_storage = next(filter(lambda registered: min_pointer >= registered.ptr and max_pointer < registered.end_ptr, self.storages), None)
    if registered_storage is None:
        raise Exception('Storage not found or pointers spanning multiple tensors')
    registered_storage.ensure_immutable()
    return registered_storage