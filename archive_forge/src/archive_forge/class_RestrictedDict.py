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
class RestrictedDict(Dict):
    """ Check for disallowed key(s).

    """

    def __init__(self, keys_type, values_type, disallow, default={}, *, help: str | None=None) -> None:
        self._disallow = set(disallow)
        super().__init__(keys_type=keys_type, values_type=values_type, default=default, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        error_keys = self._disallow & value.keys()
        if error_keys:
            msg = '' if not detail else f'Disallowed keys: {error_keys!r}'
            raise ValueError(msg)