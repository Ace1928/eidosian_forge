from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, TypeVar, Union
class IntrinsicType:
    """ Indicates usage of the intrinsic default value of a property. """

    def __copy__(self) -> IntrinsicType:
        return self

    def __str__(self) -> str:
        return 'Intrinsic'

    def __repr__(self) -> str:
        return 'Intrinsic'