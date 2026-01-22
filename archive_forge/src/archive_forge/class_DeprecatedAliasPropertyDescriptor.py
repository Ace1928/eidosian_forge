from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
class DeprecatedAliasPropertyDescriptor(AliasPropertyDescriptor[T]):
    """

    """
    alias: DeprecatedAlias[T]

    def __init__(self, name: str, alias: DeprecatedAlias[T]) -> None:
        super().__init__(name, alias)
        major, minor, patch = self.alias.since
        since = f'{major}.{minor}.{patch}'
        self.__doc__ = f'This is a backwards compatibility alias for the {self.aliased_name!r} property.\n\n.. note::\n    Property {self.name!r} was deprecated in Bokeh {since} and will be removed\n    in the future. Update your code to use {self.aliased_name!r} instead.\n'

    def _warn(self) -> None:
        deprecated(self.alias.since, self.name, self.aliased_name, self.alias.extra)

    def __get__(self, obj: HasProps | None, owner: type[HasProps] | None) -> T:
        if obj is not None:
            self._warn()
        return super().__get__(obj, owner)

    def __set__(self, obj: HasProps | None, value: T) -> None:
        self._warn()
        super().__set__(obj, value)