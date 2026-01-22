from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
class AliasPropertyDescriptor(Generic[T]):
    """

    """
    serialized: bool = False

    @property
    def aliased_name(self) -> str:
        return self.alias.aliased_name

    def __init__(self, name: str, alias: Alias[T]) -> None:
        self.name = name
        self.alias = alias
        self.property = alias
        self.__doc__ = f'This is a compatibility alias for the {self.aliased_name!r} property.'

    def __get__(self, obj: HasProps | None, owner: type[HasProps] | None) -> T:
        if obj is not None:
            return getattr(obj, self.aliased_name)
        elif owner is not None:
            return self
        raise ValueError("both 'obj' and 'owner' are None, don't know what to do")

    def __set__(self, obj: HasProps | None, value: T) -> None:
        setattr(obj, self.aliased_name, value)

    @property
    def readonly(self) -> bool:
        return self.alias.readonly

    def has_unstable_default(self, obj: HasProps) -> bool:
        return obj.lookup(self.aliased_name).has_unstable_default(obj)

    def class_default(self, cls: type[HasProps], *, no_eval: bool=False):
        return cls.lookup(self.aliased_name).class_default(cls, no_eval=no_eval)