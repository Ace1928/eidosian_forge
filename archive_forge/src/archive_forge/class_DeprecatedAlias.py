from __future__ import annotations
import logging # isort:skip
from typing import ClassVar, TypeVar
from ...util.deprecation import Version
from .bases import Property
from .descriptors import (
class DeprecatedAlias(Alias[T]):
    """
    Alias of another property of a model showing a deprecation message when used.
    """

    def __init__(self, aliased_name: str, *, since: Version, extra: str | None=None, help: str | None=None) -> None:
        super().__init__(aliased_name, help=help)
        self.since = since
        self.extra = extra

    def make_descriptors(self, base_name: str) -> list[PropertyDescriptor[T]]:
        return [DeprecatedAliasPropertyDescriptor(base_name, self)]