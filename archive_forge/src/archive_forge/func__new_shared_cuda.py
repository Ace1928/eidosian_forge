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
def _new_shared_cuda(cls, *args, **kwargs):
    return torch.UntypedStorage._new_shared_cuda(*args, **kwargs)