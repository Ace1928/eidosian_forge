from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _color_desc_256(num: int) -> str:
    """
    Return a string description of color number num.
    0..15 -> 'h0'..'h15' basic colors (as high-colors)
    16..231 -> '#000'..'#fff' color cube colors
    232..255 -> 'g3'..'g93' grays

    >>> _color_desc_256(15)
    'h15'
    >>> _color_desc_256(16)
    '#000'
    >>> _color_desc_256(17)
    '#006'
    >>> _color_desc_256(230)
    '#ffd'
    >>> _color_desc_256(233)
    'g7'
    >>> _color_desc_256(234)
    'g11'

    """
    if not 0 <= num < 256:
        raise ValueError(num)
    if num < _CUBE_START:
        return f'h{num:d}'
    if num < _GRAY_START_256:
        num -= _CUBE_START
        b, num = (num % _CUBE_SIZE_256, num // _CUBE_SIZE_256)
        g, num = (num % _CUBE_SIZE_256, num // _CUBE_SIZE_256)
        r = num % _CUBE_SIZE_256
        return f'#{_CUBE_STEPS_256_16[r]:x}{_CUBE_STEPS_256_16[g]:x}{_CUBE_STEPS_256_16[b]:x}'
    return f'g{_GRAY_STEPS_256_101[num - _GRAY_START_256]:d}'