from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar
class CollectionCheckStrategy(Enum):
    """
    Specifies how thoroughly the contents of collections are type checked.

    This has an effect on the following built-in checkers:

    * ``AbstractSet``
    * ``Dict``
    * ``List``
    * ``Mapping``
    * ``Set``
    * ``Tuple[<type>, ...]`` (arbitrarily sized tuples)

    Members:

    * ``FIRST_ITEM``: check only the first item
    * ``ALL_ITEMS``: check all items
    """
    FIRST_ITEM = auto()
    ALL_ITEMS = auto()

    def iterate_samples(self, collection: Iterable[T]) -> Iterable[T]:
        if self is CollectionCheckStrategy.FIRST_ITEM:
            try:
                return [next(iter(collection))]
            except StopIteration:
                return ()
        else:
            return collection