from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _parse_color_256(desc: str) -> int | None:
    """
    Return a color number for the description desc.
    'h0'..'h255' -> 0..255 actual color number
    '#000'..'#fff' -> 16..231 color cube colors
    'g0'..'g100' -> 16, 232..255, 231 grays and color cube black/white
    'g#00'..'g#ff' -> 16, 232...255, 231 gray and color cube black/white

    Returns None if desc is invalid.

    >>> _parse_color_256('h142')
    142
    >>> _parse_color_256('#f00')
    196
    >>> _parse_color_256('g100')
    231
    >>> _parse_color_256('g#80')
    244
    """
    if len(desc) > 4:
        return None
    try:
        if desc.startswith('h'):
            num = int(desc[1:], 10)
            if num < 0 or num > 255:
                return None
            return num
        if desc.startswith('#') and len(desc) == 4:
            rgb = int(desc[1:], 16)
            if rgb < 0:
                return None
            b, rgb = (rgb % 16, rgb // 16)
            g, r = (rgb % 16, rgb // 16)
            r = _CUBE_256_LOOKUP_16[r]
            g = _CUBE_256_LOOKUP_16[g]
            b = _CUBE_256_LOOKUP_16[b]
            return _CUBE_START + (r * _CUBE_SIZE_256 + g) * _CUBE_SIZE_256 + b
        if desc.startswith('g#'):
            gray = int(desc[2:], 16)
            if gray < 0 or gray > 255:
                return None
            gray = _GRAY_256_LOOKUP[gray]
        elif desc.startswith('g'):
            gray = int(desc[1:], 10)
            if gray < 0 or gray > 100:
                return None
            gray = _GRAY_256_LOOKUP_101[gray]
        else:
            return None
        if gray == 0:
            return _CUBE_BLACK
        gray -= 1
        if gray == _GRAY_SIZE_256:
            return _CUBE_WHITE_256
        return _GRAY_START_256 + gray
    except ValueError:
        return None