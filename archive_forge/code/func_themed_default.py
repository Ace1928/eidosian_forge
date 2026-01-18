from __future__ import annotations
import logging # isort:skip
from copy import copy
from typing import (
from ...util.dependencies import uses_pandas
from ...util.strings import nice_join
from ..has_props import HasProps
from ._sphinx import property_link, register_type_link, type_link
from .descriptor_factory import PropertyDescriptorFactory
from .descriptors import PropertyDescriptor
from .singletons import (
def themed_default(self, cls: type[HasProps], name: str, theme_overrides: dict[str, Any] | None, *, no_eval: bool=False) -> T:
    """ The default, transformed by prepare_value() and the theme overrides.

        """
    overrides = theme_overrides
    if overrides is None or name not in overrides:
        overrides = cls._overridden_defaults()
    if name in overrides:
        default = self._copy_default(overrides[name], no_eval=no_eval)
    else:
        default = self._raw_default(no_eval=no_eval)
    return self.prepare_value(cls, name, default)