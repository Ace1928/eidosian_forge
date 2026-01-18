from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
def trigger_if_changed(self, obj: HasProps, old: Any) -> None:
    """ Send a change event notification if the property is set to a
        value is not equal to ``old``.

        Args:
            obj (HasProps)
                The object the property is being set on.

            old (obj) :
                The previous value of the property to compare

        Returns:
            None

        """
    new_value = self.__get__(obj, obj.__class__)
    if not self.property.matches(old, new_value):
        self._trigger(obj, old, new_value)