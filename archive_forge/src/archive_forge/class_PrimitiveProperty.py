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
class PrimitiveProperty(Property[T]):
    """ A base class for simple property types.

    Subclasses should define a class attribute ``_underlying_type`` that is
    a tuple of acceptable type values for the property.

    Example:

        A trivial version of a ``Float`` property might look like:

        .. code-block:: python

            class Float(PrimitiveProperty):
                _underlying_type = (numbers.Real,)

    """
    _underlying_type: ClassVar[tuple[type[Any], ...]]

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if isinstance(value, self._underlying_type):
            return
        if not detail:
            raise ValueError('')
        expected_type = nice_join([cls.__name__ for cls in self._underlying_type])
        msg = f'expected a value of type {expected_type}, got {value} of type {type(value).__name__}'
        raise ValueError(msg)