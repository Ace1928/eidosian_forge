from __future__ import unicode_literals
from prompt_toolkit.filters import to_simple_filter, Condition
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from six.moves import range
import array
import errno
import os
import six
class _256ColorCache(dict):
    """
    Cach which maps (r, g, b) tuples to 256 colors.
    """

    def __init__(self):
        colors = []
        colors.append((0, 0, 0))
        colors.append((205, 0, 0))
        colors.append((0, 205, 0))
        colors.append((205, 205, 0))
        colors.append((0, 0, 238))
        colors.append((205, 0, 205))
        colors.append((0, 205, 205))
        colors.append((229, 229, 229))
        colors.append((127, 127, 127))
        colors.append((255, 0, 0))
        colors.append((0, 255, 0))
        colors.append((255, 255, 0))
        colors.append((92, 92, 255))
        colors.append((255, 0, 255))
        colors.append((0, 255, 255))
        colors.append((255, 255, 255))
        valuerange = (0, 95, 135, 175, 215, 255)
        for i in range(217):
            r = valuerange[i // 36 % 6]
            g = valuerange[i // 6 % 6]
            b = valuerange[i % 6]
            colors.append((r, g, b))
        for i in range(1, 22):
            v = 8 + i * 10
            colors.append((v, v, v))
        self.colors = colors

    def __missing__(self, value):
        r, g, b = value
        distance = 257 * 257 * 3
        match = 0
        for i, (r2, g2, b2) in enumerate(self.colors):
            d = (r - r2) ** 2 + (g - g2) ** 2 + (b - b2) ** 2
            if d < distance:
                match = i
                distance = d
        self[value] = match
        return match