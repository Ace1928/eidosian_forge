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
def _get_closest_ansi_color(r, g, b, exclude=()):
    """
    Find closest ANSI color. Return it by name.

    :param r: Red (Between 0 and 255.)
    :param g: Green (Between 0 and 255.)
    :param b: Blue (Between 0 and 255.)
    :param exclude: A tuple of color names to exclude. (E.g. ``('ansired', )``.)
    """
    assert isinstance(exclude, tuple)
    saturation = abs(r - g) + abs(g - b) + abs(b - r)
    if saturation > 30:
        exclude += ('ansilightgray', 'ansidarkgray', 'ansiwhite', 'ansiblack')
    distance = 257 * 257 * 3
    match = 'ansidefault'
    for name, (r2, g2, b2) in ANSI_COLORS_TO_RGB.items():
        if name != 'ansidefault' and name not in exclude:
            d = (r - r2) ** 2 + (g - g2) ** 2 + (b - b2) ** 2
            if d < distance:
                match = name
                distance = d
    return match