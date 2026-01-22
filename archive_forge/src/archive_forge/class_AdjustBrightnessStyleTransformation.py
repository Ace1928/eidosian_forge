from __future__ import annotations
from abc import ABCMeta, abstractmethod
from colorsys import hls_to_rgb, rgb_to_hls
from typing import Callable, Hashable, Sequence
from prompt_toolkit.cache import memoized
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.utils import AnyFloat, to_float, to_str
from .base import ANSI_COLOR_NAMES, Attrs
from .style import parse_color
class AdjustBrightnessStyleTransformation(StyleTransformation):
    """
    Adjust the brightness to improve the rendering on either dark or light
    backgrounds.

    For dark backgrounds, it's best to increase `min_brightness`. For light
    backgrounds it's best to decrease `max_brightness`. Usually, only one
    setting is adjusted.

    This will only change the brightness for text that has a foreground color
    defined, but no background color. It works best for 256 or true color
    output.

    .. note:: Notice that there is no universal way to detect whether the
              application is running in a light or dark terminal. As a
              developer of an command line application, you'll have to make
              this configurable for the user.

    :param min_brightness: Float between 0.0 and 1.0 or a callable that returns
        a float.
    :param max_brightness: Float between 0.0 and 1.0 or a callable that returns
        a float.
    """

    def __init__(self, min_brightness: AnyFloat=0.0, max_brightness: AnyFloat=1.0) -> None:
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def transform_attrs(self, attrs: Attrs) -> Attrs:
        min_brightness = to_float(self.min_brightness)
        max_brightness = to_float(self.max_brightness)
        assert 0 <= min_brightness <= 1
        assert 0 <= max_brightness <= 1
        if min_brightness == 0.0 and max_brightness == 1.0:
            return attrs
        no_background = not attrs.bgcolor or attrs.bgcolor == 'default'
        has_fgcolor = attrs.color and attrs.color != 'ansidefault'
        if has_fgcolor and no_background:
            r, g, b = self._color_to_rgb(attrs.color or '')
            hue, brightness, saturation = rgb_to_hls(r, g, b)
            brightness = self._interpolate_brightness(brightness, min_brightness, max_brightness)
            r, g, b = hls_to_rgb(hue, brightness, saturation)
            new_color = f'{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
            attrs = attrs._replace(color=new_color)
        return attrs

    def _color_to_rgb(self, color: str) -> tuple[float, float, float]:
        """
        Parse `style.Attrs` color into RGB tuple.
        """
        try:
            from prompt_toolkit.output.vt100 import ANSI_COLORS_TO_RGB
            r, g, b = ANSI_COLORS_TO_RGB[color]
            return (r / 255.0, g / 255.0, b / 255.0)
        except KeyError:
            pass
        return (int(color[0:2], 16) / 255.0, int(color[2:4], 16) / 255.0, int(color[4:6], 16) / 255.0)

    def _interpolate_brightness(self, value: float, min_brightness: float, max_brightness: float) -> float:
        """
        Map the brightness to the (min_brightness..max_brightness) range.
        """
        return min_brightness + (max_brightness - min_brightness) * value

    def invalidation_hash(self) -> Hashable:
        return ('adjust-brightness', to_float(self.min_brightness), to_float(self.max_brightness))