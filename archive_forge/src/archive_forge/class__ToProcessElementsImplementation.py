import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
class _ToProcessElementsImplementation(Generic[_T]):
    """Auxiliary data class for LinearBase._flatten_once_and_add_to()."""
    __slots__ = ('_queue',)

    def __init__(self, term: _T, scale: float) -> None:
        self._queue: Deque[Tuple[_T, float]] = collections.deque([(term, scale)])

    def append(self, term: _T, scale: float) -> None:
        self._queue.append((term, scale))

    def pop(self) -> Tuple[_T, float]:
        return self._queue.popleft()

    def __bool__(self) -> bool:
        return bool(self._queue)