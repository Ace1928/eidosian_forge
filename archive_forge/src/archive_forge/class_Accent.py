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
class Accent(Char):
    """
    The font metrics need to be dealt with differently for accents,
    since they are already offset correctly from the baseline in
    TrueType fonts.
    """

    def _update_metrics(self) -> None:
        metrics = self._metrics = self.fontset.get_metrics(self.font, self.font_class, self.c, self.fontsize, self.dpi)
        self.width = metrics.xmax - metrics.xmin
        self.height = metrics.ymax - metrics.ymin
        self.depth = 0

    def shrink(self) -> None:
        super().shrink()
        self._update_metrics()

    def render(self, output: Output, x: float, y: float) -> None:
        self.fontset.render_glyph(output, x - self._metrics.xmin, y + self._metrics.ymin, self.font, self.font_class, self.c, self.fontsize, self.dpi)