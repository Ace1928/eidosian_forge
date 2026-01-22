from __future__ import annotations
import logging # isort:skip
from collections.abc import (
from typing import TYPE_CHECKING, Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .descriptors import ColumnDataPropertyDescriptor
from .enum import Enum
from .numeric import Int
from .singletons import Intrinsic, Undefined
from .wrappers import (
class RelativeDelta(Dict):
    """ Accept RelativeDelta dicts for time delta values.

    """

    def __init__(self, default={}, *, help: str | None=None) -> None:
        keys = Enum('years', 'months', 'days', 'hours', 'minutes', 'seconds', 'microseconds')
        values = Int
        super().__init__(keys, values, default=default, help=help)

    def __str__(self) -> str:
        return self.__class__.__name__