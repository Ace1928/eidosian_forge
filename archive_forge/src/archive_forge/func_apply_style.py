from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
@classmethod
def apply_style(cls, segments: Iterable['Segment'], style: Optional[Style]=None, post_style: Optional[Style]=None) -> Iterable['Segment']:
    """Apply style(s) to an iterable of segments.

        Returns an iterable of segments where the style is replaced by ``style + segment.style + post_style``.

        Args:
            segments (Iterable[Segment]): Segments to process.
            style (Style, optional): Base style. Defaults to None.
            post_style (Style, optional): Style to apply on top of segment style. Defaults to None.

        Returns:
            Iterable[Segments]: A new iterable of segments (possibly the same iterable).
        """
    result_segments = segments
    if style:
        apply = style.__add__
        result_segments = (cls(text, None if control else apply(_style), control) for text, _style, control in result_segments)
    if post_style:
        result_segments = (cls(text, None if control else _style + post_style if _style else post_style, control) for text, _style, control in result_segments)
    return result_segments