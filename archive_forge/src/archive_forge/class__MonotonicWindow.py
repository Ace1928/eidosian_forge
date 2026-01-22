from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
class _MonotonicWindow(List[T]):
    """Custom fixed size list that only supports right-append and ensures that all values increase monotonically."""

    def __init__(self, maxlen: int) -> None:
        super().__init__()
        self.maxlen = maxlen

    @property
    def last(self) -> Optional[T]:
        if len(self) > 0:
            return self[-1]
        return None

    @override
    def append(self, x: T) -> None:
        last = self.last
        if last is not None and last >= x:
            raise ValueError(f'Expected the value to increase, last: {last}, current: {x}')
        list.append(self, x)
        if len(self) > self.maxlen:
            del self[0]

    @override
    def __setitem__(self, key: Any, value: Any) -> None:
        raise NotImplementedError('__setitem__ is not supported')