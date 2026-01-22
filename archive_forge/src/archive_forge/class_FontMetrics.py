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
class FontMetrics(NamedTuple):
    """
    Metrics of a font.

    Attributes
    ----------
    advance : float
        The advance distance (in points) of the glyph.
    height : float
        The height of the glyph in points.
    width : float
        The width of the glyph in points.
    xmin, xmax, ymin, ymax : float
        The ink rectangle of the glyph.
    iceberg : float
        The distance from the baseline to the top of the glyph. (This corresponds to
        TeX's definition of "height".)
    slanted : bool
        Whether the glyph should be considered as "slanted" (currently used for kerning
        sub/superscripts).
    """
    advance: float
    height: float
    width: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    iceberg: float
    slanted: bool