from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ...util.dataclasses import Unspecified
from ...util.serialization import convert_datetime_type, convert_timedelta_type
from ...util.strings import nice_join
from .. import enums
from .color import ALPHA_DEFAULT_HELP, COLOR_DEFAULT_HELP, Color
from .datetime import Datetime, TimeDelta
from .descriptors import DataSpecPropertyDescriptor, UnitsSpecPropertyDescriptor
from .either import Either
from .enum import Enum
from .instance import Instance
from .nothing import Nothing
from .nullable import Nullable
from .primitive import (
from .serialized import NotSerialized
from .singletons import Undefined
from .struct import Optional, Struct
from .vectorization import (
from .visual import (
class UnitsSpec(NumberSpec):
    """ A |DataSpec| property that accepts numeric fixed values, and also
    provides an associated units property to store units information.

    """

    def __init__(self, default, units_enum, units_default, *, help: str | None=None) -> None:
        super().__init__(default=default, help=help)
        units_type = NotSerialized(Enum(units_enum), default=units_default, help=f'\n        Units to use for the associated property: {nice_join(units_enum)}\n        ')
        self._units_type = self._validate_type_param(units_type, help_allowed=True)
        self._type_params += [Struct(value=self.value_type, transform=Optional(Instance('bokeh.models.transforms.Transform')), units=Optional(units_type)), Struct(field=String, transform=Optional(Instance('bokeh.models.transforms.Transform')), units=Optional(units_type)), Struct(expr=Instance('bokeh.models.expressions.Expression'), transform=Optional(Instance('bokeh.models.transforms.Transform')), units=Optional(units_type))]

    def __str__(self) -> str:
        units_default = self._units_type._default
        return f'{self.__class__.__name__}(units_default={units_default!r})'

    def get_units(self, obj: HasProps, name: str) -> str:
        return getattr(obj, name + '_units')

    def make_descriptors(self, base_name: str):
        """ Return a list of ``PropertyDescriptor`` instances to install on a
        class, in order to delegate attribute access to this property.

        Unlike simpler property types, ``UnitsSpec`` returns multiple
        descriptors to install. In particular, descriptors for the base
        property as well as the associated units property are returned.

        Args:
            name (str) : the name of the property these descriptors are for

        Returns:
            list[PropertyDescriptor]

        The descriptors returned are collected by the ``MetaHasProps``
        metaclass and added to ``HasProps`` subclasses during class creation.
        """
        units_name = base_name + '_units'
        units_props = self._units_type.make_descriptors(units_name)
        return [*units_props, UnitsSpecPropertyDescriptor(base_name, self, units_props[0])]

    def to_serializable(self, obj: HasProps, name: str, val: Any) -> Vectorized:
        val = super().to_serializable(obj, name, val)
        if val.units is Unspecified:
            units = self.get_units(obj, name)
            if units != self._units_type._default:
                val.units = units
        return val