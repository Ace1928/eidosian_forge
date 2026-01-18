import sys
from functools import lru_cache
from marshal import dumps, loads
from random import randint
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast
from . import errors
from .color import Color, ColorParseError, ColorSystem, blend_rgb
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME, TerminalTheme
@property
def without_color(self) -> 'Style':
    """Get a copy of the style with color removed."""
    if self._null:
        return NULL_STYLE
    style: Style = self.__new__(Style)
    style._ansi = None
    style._style_definition = None
    style._color = None
    style._bgcolor = None
    style._attributes = self._attributes
    style._set_attributes = self._set_attributes
    style._link = self._link
    style._link_id = f'{randint(0, 999999)}' if self._link else ''
    style._null = False
    style._meta = None
    style._hash = None
    return style