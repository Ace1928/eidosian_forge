from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
def _get_default(self, obj: HasProps) -> T:
    """ Internal implementation of instance attribute access for default
        values.

        Handles bookeeping around ``PropertyContainer`` value, etc.

        """
    if self.name in obj._property_values:
        raise RuntimeError('Bokeh internal error, does not handle the case of self.name already in _property_values')
    themed_values = obj.themed_values()
    is_themed = themed_values is not None and self.name in themed_values
    unstable_dict = obj._unstable_themed_values if is_themed else obj._unstable_default_values
    if self.name in unstable_dict:
        return unstable_dict[self.name]
    default = self.instance_default(obj)
    if self.has_unstable_default(obj):
        if isinstance(default, PropertyValueContainer):
            default._register_owner(obj, self)
        unstable_dict[self.name] = default
    return default