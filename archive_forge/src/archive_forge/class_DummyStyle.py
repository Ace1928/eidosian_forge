from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Callable, Hashable, NamedTuple
class DummyStyle(BaseStyle):
    """
    A style that doesn't style anything.
    """

    def get_attrs_for_style_str(self, style_str: str, default: Attrs=DEFAULT_ATTRS) -> Attrs:
        return default

    def invalidation_hash(self) -> Hashable:
        return 1

    @property
    def style_rules(self) -> list[tuple[str, str]]:
        return []