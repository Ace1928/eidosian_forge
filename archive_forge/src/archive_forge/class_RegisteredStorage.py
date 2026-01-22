from __future__ import annotations
import dataclasses
from . import torch_wrapper
@dataclasses.dataclass
class RegisteredStorage:
    storage: torch.Storage
    dtype: torch.dtype
    size: int
    ptr: int

    @property
    def end_ptr(self) -> int:
        return self.ptr + self.size

    @property
    def access_tensor(self) -> torch.Tensor:
        return torch.tensor(self.storage, dtype=self.dtype, device=self.storage.device)

    def ensure_immutable(self):
        assert self.storage.data_ptr() == self.ptr and self.storage.size() == self.size