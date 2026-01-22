from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _color_desc_88(num: int) -> str:
    """
    Return a string description of color number num.
    0..15 -> 'h0'..'h15' basic colors (as high-colors)
    16..79 -> '#000'..'#fff' color cube colors
    80..87 -> 'g18'..'g90' grays

    >>> _color_desc_88(15)
    'h15'
    >>> _color_desc_88(16)
    '#000'
    >>> _color_desc_88(17)
    '#008'
    >>> _color_desc_88(78)
    '#ffc'
    >>> _color_desc_88(81)
    'g36'
    >>> _color_desc_88(82)
    'g45'

    """
    if not 0 < num < 88:
        raise ValueError(num)
    if num < _CUBE_START:
        return f'h{num:d}'
    if num < _GRAY_START_88:
        num -= _CUBE_START
        b, num = (num % _CUBE_SIZE_88, num // _CUBE_SIZE_88)
        g, r = (num % _CUBE_SIZE_88, num // _CUBE_SIZE_88)
        return f'#{_CUBE_STEPS_88_16[r]:x}{_CUBE_STEPS_88_16[g]:x}{_CUBE_STEPS_88_16[b]:x}'
    return f'g{_GRAY_STEPS_88_101[num - _GRAY_START_88]:d}'