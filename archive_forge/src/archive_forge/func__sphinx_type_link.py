from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar
from ...util.strings import nice_join
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .singletons import Intrinsic
@register_type_link(Either)
def _sphinx_type_link(obj: Either[Any]):
    subtypes = ', '.join((type_link(x) for x in obj.type_params))
    return f'{property_link(obj)}({subtypes})'