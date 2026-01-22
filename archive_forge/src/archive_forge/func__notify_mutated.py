from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
def _notify_mutated(self, obj: HasProps, old: Any, hint: DocumentPatchedEvent | None=None) -> None:
    """ A method to call when a container is mutated "behind our back"
        and we detect it with our ``PropertyContainer`` wrappers.

        Args:
            obj (HasProps) :
                The object who's container value was mutated

            old (object) :
                The "old" value of the container

                In this case, somewhat weirdly, ``old`` is a copy and the
                new value should already be set unless we change it due to
                validation.

            hint (event hint or None, optional)
                An optional update event hint, e.g. ``ColumnStreamedEvent``
                (default: None)

                Update event hints are usually used at times when better
                update performance can be obtained by special-casing in
                some way (e.g. streaming or patching column data sources)

        Returns:
            None

        """
    value = self.__get__(obj, obj.__class__)
    value = self.property.prepare_value(obj, self.name, value, hint=hint)
    self._set(obj, old, value, hint=hint)