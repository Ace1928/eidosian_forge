from __future__ import annotations
import dataclasses
from . import torch_wrapper
def add_tensor(self, t: torch.Tensor):
    storage = t.untyped_storage()
    self.storages.append(RegisteredStorage(storage, t.dtype, storage.size(), storage.data_ptr()))
    return t.data_ptr()