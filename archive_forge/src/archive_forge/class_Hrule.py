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
class Hrule(Rule):
    """Convenience class to create a horizontal rule."""

    def __init__(self, state: ParserState, thickness: float | None=None):
        if thickness is None:
            thickness = state.get_current_underline_thickness()
        height = depth = thickness * 0.5
        super().__init__(np.inf, height, depth, state)