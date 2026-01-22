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
def _new_shared(cls, size):
    """Create a new storage in shared memory with the same data type."""
    untyped_storage = torch.UntypedStorage._new_shared(size * cls()._element_size())
    return cls(wrap_storage=untyped_storage)