from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c: str, height: float, depth: float, state: ParserState, always: bool=False, factor: float | None=None):
        alternatives = state.fontset.get_sized_alternatives_for_symbol(state.font, c)
        xHeight = state.fontset.get_xheight(state.font, state.fontsize, state.dpi)
        state = state.copy()
        target_total = height + depth
        for fontname, sym in alternatives:
            state.font = fontname
            char = Char(sym, state)
            if char.height + char.depth >= target_total - 0.2 * xHeight:
                break
        shift = 0.0
        if state.font != 0 or len(alternatives) == 1:
            if factor is None:
                factor = target_total / (char.height + char.depth)
            state.fontsize *= factor
            char = Char(sym, state)
            shift = depth - char.depth
        super().__init__([char])
        self.shift_amount = shift