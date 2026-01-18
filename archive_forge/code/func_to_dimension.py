from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Union
def to_dimension(value: AnyDimension) -> Dimension:
    """
    Turn the given object into a `Dimension` object.
    """
    if value is None:
        return Dimension()
    if isinstance(value, int):
        return Dimension.exact(value)
    if isinstance(value, Dimension):
        return value
    if callable(value):
        return to_dimension(value())
    raise ValueError('Not an integer or Dimension object.')