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
def _warn_typed_storage_removal(stacklevel=2):
    global _always_warn_typed_storage_removal

    def is_first_time():
        if not hasattr(_warn_typed_storage_removal, 'has_warned'):
            return True
        else:
            return not _warn_typed_storage_removal.__dict__['has_warned']
    if _get_always_warn_typed_storage_removal() or is_first_time():
        message = 'TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()'
        warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
        _warn_typed_storage_removal.__dict__['has_warned'] = True