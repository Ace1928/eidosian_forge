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
@classmethod
def _overridden_defaults(cls) -> dict[str, Any]:
    """ Returns a dictionary of defaults that have been overridden.

        .. note::
            This is an implementation detail of ``Property``.

        """
    defaults: dict[str, Any] = {}
    for c in reversed(cls.__mro__):
        defaults.update(getattr(c, '__overridden_defaults__', {}))
    return defaults