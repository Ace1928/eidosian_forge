from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def __set_background(self, background: str) -> None:
    flags = 0
    if background in {'', 'default'}:
        color = 0
    elif background in _BASIC_COLORS:
        color = _BASIC_COLORS.index(background)
        flags |= _BG_BASIC_COLOR
    elif self.__value & _HIGH_88_COLOR:
        color = _parse_color_88(background)
        flags |= _BG_HIGH_COLOR
    elif self.__value & _HIGH_TRUE_COLOR:
        color = _parse_color_true(background)
        flags |= _BG_TRUE_COLOR
    else:
        color = _parse_color_256(_true_to_256(background) or background)
        flags |= _BG_HIGH_COLOR
    if color is None:
        raise AttrSpecError(f'Unrecognised color specification in background ({background!r})')
    self.__value = self.__value & ~_BG_MASK | color << _BG_SHIFT | flags