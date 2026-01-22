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
def _share_filename_cpu_(self, *args, **kwargs):
    manager_handle, storage_handle, size = self._untyped_storage._share_filename_cpu_(*args, **kwargs)
    return (manager_handle, storage_handle, size // self._element_size())