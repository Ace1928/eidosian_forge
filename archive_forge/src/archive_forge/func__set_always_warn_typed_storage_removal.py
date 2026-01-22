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
def _set_always_warn_typed_storage_removal(always_warn):
    global _always_warn_typed_storage_removal
    assert isinstance(always_warn, bool)
    _always_warn_typed_storage_removal = always_warn