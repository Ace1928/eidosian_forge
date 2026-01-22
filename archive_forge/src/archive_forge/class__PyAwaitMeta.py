from __future__ import annotations
from typing import cast, Callable, Generic, Type, TypeVar
import torch
class _PyAwaitMeta(type(torch._C._Await), type(Generic)):
    pass