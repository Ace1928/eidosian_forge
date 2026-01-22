import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
@dataclasses.dataclass(frozen=True)
class QuadraticTermIdKey:
    """An ordered pair of ints used as a key for quadratic terms.

    QuadraticTermIdKey.id1 <= QuadraticTermIdKey.id2.
    """
    __slots__ = ('id1', 'id2')
    id1: int
    id2: int

    def __init__(self, a: int, b: int):
        """Ints a and b will be ordered internally."""
        id1 = a
        id2 = b
        if id1 > id2:
            id1, id2 = (id2, id1)
        object.__setattr__(self, 'id1', id1)
        object.__setattr__(self, 'id2', id2)