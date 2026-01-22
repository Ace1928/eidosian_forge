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
@register_type_link(Tuple)
def _sphinx_type_tuple(obj: Tuple):
    item_types = ', '.join((type_link(x) for x in obj.type_params))
    return f'{property_link(obj)}({item_types})'