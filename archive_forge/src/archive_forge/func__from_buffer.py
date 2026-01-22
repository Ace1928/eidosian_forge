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
@classmethod
def _from_buffer(cls, *args, dtype=None, device=None, **kwargs):
    if cls == TypedStorage:
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device('cpu' if device is None else device)
        if device.type != 'cpu':
            raise RuntimeError(f'TypedStorage.from_buffer: Not available for device {device.type}')
        untyped_storage: torch.UntypedStorage = torch.UntypedStorage.from_buffer(*args, dtype=dtype, **kwargs)
    else:
        if dtype is not None or len(args) == 5:
            raise RuntimeError("from_buffer: 'dtype' can only be specified in UntypedStorage.from_buffer and TypedStorage.from_buffer")
        if device is not None:
            raise RuntimeError("from_buffer: 'device' can only be specified in UntypedStorage.from_buffer and TypedStorage.from_buffer")
        dtype = cls._dtype
        untyped_storage = torch.UntypedStorage.from_buffer(*args, dtype=dtype, **kwargs)
    return TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)