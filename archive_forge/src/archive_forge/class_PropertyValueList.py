from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
class PropertyValueList(PropertyValueContainer, list[T]):
    """ A list property value container that supports change notifications on
    mutating operations.

    When a Bokeh model has a ``List`` property, the ``PropertyValueLists`` are
    transparently created to wrap those values. These ``PropertyValueList``
    values are subject to normal property validation. If the property type
    ``foo = List(Str)`` then attempting to set ``x.foo[0] = 10`` will raise
    an error.

    Instances of ``PropertyValueList`` can be explicitly created by passing
    any object that the standard list initializer accepts, for example:

    .. code-block:: python

        >>> PropertyValueList([10, 20])
        [10, 20]

        >>> PropertyValueList((10, 20))
        [10, 20]

    The following mutating operations on lists automatically trigger
    notifications:

    .. code-block:: python

        del x[y]
        del x[i:j]
        x += y
        x *= y
        x[i] = y
        x[i:j] = y
        x.append
        x.extend
        x.insert
        x.pop
        x.remove
        x.reverse
        x.sort

    """

    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def _saved_copy(self) -> list[T]:
        return list(self)

    @notify_owner
    def __delitem__(self, y):
        return super().__delitem__(y)

    @notify_owner
    def __iadd__(self, y):
        return super().__iadd__(y)

    @notify_owner
    def __imul__(self, y):
        return super().__imul__(y)

    @notify_owner
    def __setitem__(self, i, y):
        return super().__setitem__(i, y)

    @notify_owner
    def append(self, obj):
        return super().append(obj)

    @notify_owner
    def extend(self, iterable):
        return super().extend(iterable)

    @notify_owner
    def insert(self, index, obj):
        return super().insert(index, obj)

    @notify_owner
    def pop(self, index=-1):
        return super().pop(index)

    @notify_owner
    def remove(self, obj):
        return super().remove(obj)

    @notify_owner
    def reverse(self):
        return super().reverse()

    @notify_owner
    def sort(self, **kwargs):
        return super().sort(**kwargs)