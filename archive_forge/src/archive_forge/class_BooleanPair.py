import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
class BooleanPair(Pair):
    """Pair for :class:`bool` inputs.

    .. note::

        If ``numpy`` is available, also handles :class:`numpy.bool_` inputs.

    """

    def __init__(self, actual: Any, expected: Any, *, id: Tuple[Any, ...], **other_parameters: Any) -> None:
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, **other_parameters)

    @property
    def _supported_types(self) -> Tuple[Type, ...]:
        cls: List[Type] = [bool]
        if NUMPY_AVAILABLE:
            cls.append(np.bool_)
        return tuple(cls)

    def _process_inputs(self, actual: Any, expected: Any, *, id: Tuple[Any, ...]) -> Tuple[bool, bool]:
        self._check_inputs_isinstance(actual, expected, cls=self._supported_types)
        actual, expected = (self._to_bool(bool_like, id=id) for bool_like in (actual, expected))
        return (actual, expected)

    def _to_bool(self, bool_like: Any, *, id: Tuple[Any, ...]) -> bool:
        if isinstance(bool_like, bool):
            return bool_like
        elif isinstance(bool_like, np.bool_):
            return bool_like.item()
        else:
            raise ErrorMeta(TypeError, f'Unknown boolean type {type(bool_like)}.', id=id)

    def compare(self) -> None:
        if self.actual is not self.expected:
            self._fail(AssertionError, f'Booleans mismatch: {self.actual} is not {self.expected}')