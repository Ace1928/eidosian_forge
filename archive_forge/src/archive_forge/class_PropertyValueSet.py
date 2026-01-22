from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
class PropertyValueSet(PropertyValueContainer, set[T]):
    """ A list property value container that supports change notifications on
    mutating operations.

    """

    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def _saved_copy(self) -> set[T]:
        return set(self)

    @notify_owner
    def add(self, element: T) -> None:
        super().add(element)

    @notify_owner
    def difference_update(self, *s: Iterable[Any]) -> None:
        super().difference_update(*s)

    @notify_owner
    def discard(self, element: T) -> None:
        super().discard(element)

    @notify_owner
    def intersection_update(self, *s: Iterable[Any]) -> None:
        super().intersection_update(*s)

    @notify_owner
    def remove(self, element: T) -> None:
        super().discard(element)

    @notify_owner
    def symmetric_difference_update(self, s: Iterable[T]) -> None:
        super().symmetric_difference_update(s)

    @notify_owner
    def update(self, *s: Iterable[T]) -> None:
        super().update(*s)