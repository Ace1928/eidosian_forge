from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
class PropertyValueContainer:
    """ A base class for property container classes that support change
    notifications on mutating operations.

    This class maintains an internal list of property owners, and also
    provides a private mechanism for methods wrapped with
    :func:`~bokeh.core.property.wrappers.notify_owners` to update
    those owners when mutating changes occur.

    """
    _owners: set[tuple[HasProps, PropertyDescriptor[Any]]]

    def __init__(self, *args, **kwargs) -> None:
        self._owners = set()
        super().__init__(*args, **kwargs)

    def _register_owner(self, owner: HasProps, descriptor: PropertyDescriptor[Any]) -> None:
        self._owners.add((owner, descriptor))

    def _unregister_owner(self, owner: HasProps, descriptor: PropertyDescriptor[Any]) -> None:
        self._owners.discard((owner, descriptor))

    def _notify_owners(self, old: Any, hint: DocumentPatchedEvent | None=None) -> None:
        for owner, descriptor in self._owners:
            descriptor._notify_mutated(owner, old, hint=hint)

    def _saved_copy(self) -> Any:
        raise RuntimeError('Subtypes must implement this to make a backup copy')