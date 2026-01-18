from __future__ import annotations
import logging # isort:skip
import difflib
from typing import (
from weakref import WeakSet
from ..settings import settings
from ..util.strings import append_docstring, nice_join
from ..util.warnings import warn
from .property.descriptor_factory import PropertyDescriptorFactory
from .property.descriptors import PropertyDescriptor, UnsetValueError
from .property.override import Override
from .property.singletons import Intrinsic, Undefined
from .property.wrappers import PropertyValueContainer
from .serialization import (
from .types import ID
def themed_values(self) -> dict[str, Any] | None:
    """ Get any theme-provided overrides.

        Results are returned as a dict from property name to value, or
        ``None`` if no theme overrides any values for this instance.

        Returns:
            dict or None

        """
    return getattr(self, '__themed_values__', None)