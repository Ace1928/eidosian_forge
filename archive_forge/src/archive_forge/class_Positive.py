from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar, Union
from ...util.deprecation import deprecated
from .bases import (
from .primitive import Float, Int
from .singletons import Intrinsic, Undefined
class Positive(SingleParameterizedProperty[T]):
    """ A property accepting a value of some other type while having undefined default. """

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T]=Intrinsic, help: str | None=None) -> None:
        super().__init__(type_param, default=default, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if not 0 < value:
            raise ValueError(f'expected a positive number, got {value!r}')