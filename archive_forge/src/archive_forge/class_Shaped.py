from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol, Union, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike, NDArray
@runtime_checkable
class Shaped(Protocol):
    """Protocol that defines what it means to be a shaped object.

    Note that static type checkers will classify ``numpy.ndarray`` as being :class:`Shaped`.
    Moreover, since this protocol is runtime-checkable, we will even have
    ``isinstance(<numpy.ndarray instance>, Shaped) == True``.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of this object."""
        raise NotImplementedError('A `Shaped` protocol must implement the `shape` property')

    @property
    def ndim(self) -> int:
        """The number of array dimensions of this object."""
        raise NotImplementedError('A `Shaped` protocol must implement the `ndim` property')

    @property
    def size(self) -> int:
        """The total dimension of this object, i.e. the product of the entries of :attr:`~shape`."""
        raise NotImplementedError('A `Shaped` protocol must implement the `size` property')