from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
class SelectorPattern:
    """Selector pattern."""

    def __init__(self, name: str, pattern: str) -> None:
        """Initialize."""
        self.name = name
        self.re_pattern = re.compile(pattern, re.I | re.X | re.U)

    def get_name(self) -> str:
        """Get name."""
        return self.name

    def match(self, selector: str, index: int, flags: int) -> Match[str] | None:
        """Match the selector."""
        return self.re_pattern.match(selector, index)