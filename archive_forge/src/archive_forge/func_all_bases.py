from __future__ import annotations
from typing import TYPE_CHECKING, cast
from more_itertools import unique_everseen
def all_bases(c: type[object]) -> list[type[Any]]:
    """
    return a tuple of all base classes the class c has as a parent.
    >>> object in all_bases(list)
    True
    """
    return c.mro()[1:]