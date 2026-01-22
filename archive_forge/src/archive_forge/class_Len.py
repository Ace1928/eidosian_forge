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
class Len(SingleParameterizedProperty[TSeq]):
    """ Allows only containers of the given length. """

    def __init__(self, type_param: TypeOrInst[TSeq], length: int, *, default: Init[TSeq]=Intrinsic, help: str | None=None) -> None:
        super().__init__(type_param, default=default, help=help)
        self.length = length

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if len(value) != self.length:
            msg = '' if not detail else f'Expected a container of length #{self.length}, got #{len(value)}'
            raise ValueError(msg)