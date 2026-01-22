from __future__ import annotations
import logging # isort:skip
from typing import TypeVar
from .bases import (
from .singletons import Undefined
class Readonly(SingleParameterizedProperty[T]):
    """ A property that can't be manually modified by the user. """
    _readonly = True

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T]=Undefined, help: str | None=None) -> None:
        super().__init__(type_param, default=default, help=help)