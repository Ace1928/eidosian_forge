import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
class _LegacyStorage(TypedStorage, metaclass=_LegacyStorageMeta):

    @classmethod
    def _new_shared(cls, size):
        """Create a new storage in shared memory with the same data type."""
        untyped_storage = torch.UntypedStorage._new_shared(size * cls()._element_size())
        return cls(wrap_storage=untyped_storage)

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _new_shared_filename(cls, manager, obj, size):
        bytes_size = size * torch._utils._element_size(cls.dtype)
        return cls(wrap_storage=torch.UntypedStorage._new_shared_filename_cpu(manager, obj, bytes_size))