from __future__ import annotations
import itertools
import re
from enum import Enum
from typing import Hashable, TypeVar
from prompt_toolkit.cache import SimpleCache
from .base import (
from .named_colors import NAMED_COLORS
class _MergedStyle(BaseStyle):
    """
    Merge multiple `Style` objects into one.
    This is supposed to ensure consistency: if any of the given styles changes,
    then this style will be updated.
    """

    def __init__(self, styles: list[BaseStyle]) -> None:
        self.styles = styles
        self._style: SimpleCache[Hashable, Style] = SimpleCache(maxsize=1)

    @property
    def _merged_style(self) -> Style:
        """The `Style` object that has the other styles merged together."""

        def get() -> Style:
            return Style(self.style_rules)
        return self._style.get(self.invalidation_hash(), get)

    @property
    def style_rules(self) -> list[tuple[str, str]]:
        style_rules = []
        for s in self.styles:
            style_rules.extend(s.style_rules)
        return style_rules

    def get_attrs_for_style_str(self, style_str: str, default: Attrs=DEFAULT_ATTRS) -> Attrs:
        return self._merged_style.get_attrs_for_style_str(style_str, default)

    def invalidation_hash(self) -> Hashable:
        return tuple((s.invalidation_hash() for s in self.styles))