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
def _genset(self, s: str, loc: int, toks: ParseResults) -> T.Any:
    annotation = toks['annotation']
    body = toks['body']
    thickness = self.get_state().get_current_underline_thickness()
    annotation.shrink()
    cannotation = HCentered([annotation])
    cbody = HCentered([body])
    width = max(cannotation.width, cbody.width)
    cannotation.hpack(width, 'exactly')
    cbody.hpack(width, 'exactly')
    vgap = thickness * 3
    if s[loc + 1] == 'u':
        vlist = Vlist([cbody, Vbox(0, vgap), cannotation])
        vlist.shift_amount = cbody.depth + cannotation.height + vgap
    else:
        vlist = Vlist([cannotation, Vbox(0, vgap), cbody])
    return vlist