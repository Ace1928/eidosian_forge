from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
class StyleTransformation(metaclass=ABCMeta):
    """
    Base class for any style transformation.
    """

    @abstractmethod
    def transform_attrs(self, attrs: Attrs) -> Attrs:
        """
        Take an `Attrs` object and return a new `Attrs` object.

        Remember that the color formats can be either "ansi..." or a 6 digit
        lowercase hexadecimal color (without '#' prefix).
        """

    def invalidation_hash(self) -> Hashable:
        """
        When this changes, the cache should be invalidated.
        """
        return f'{self.__class__.__name__}-{id(self)}'