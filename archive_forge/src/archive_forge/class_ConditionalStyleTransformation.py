from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
class ConditionalStyleTransformation(StyleTransformation):
    """
    Apply the style transformation depending on a condition.
    """

    def __init__(self, style_transformation: StyleTransformation, filter: FilterOrBool) -> None:
        self.style_transformation = style_transformation
        self.filter = to_filter(filter)

    def transform_attrs(self, attrs: Attrs) -> Attrs:
        if self.filter():
            return self.style_transformation.transform_attrs(attrs)
        return attrs

    def invalidation_hash(self) -> Hashable:
        return (self.filter(), self.style_transformation.invalidation_hash())