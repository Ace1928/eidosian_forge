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
def _map_virtual_font(self, fontname: str, font_class: str, uniindex: int) -> tuple[str, int]:
    font_mapping = stix_virtual_fonts.get(fontname)
    if self._sans and font_mapping is None and (fontname not in ('regular', 'default')):
        font_mapping = stix_virtual_fonts['sf']
        doing_sans_conversion = True
    else:
        doing_sans_conversion = False
    if isinstance(font_mapping, dict):
        try:
            mapping = font_mapping[font_class]
        except KeyError:
            mapping = font_mapping['rm']
    elif isinstance(font_mapping, list):
        mapping = font_mapping
    else:
        mapping = None
    if mapping is not None:
        lo = 0
        hi = len(mapping)
        while lo < hi:
            mid = (lo + hi) // 2
            range = mapping[mid]
            if uniindex < range[0]:
                hi = mid
            elif uniindex <= range[1]:
                break
            else:
                lo = mid + 1
        if range[0] <= uniindex <= range[1]:
            uniindex = uniindex - range[0] + range[3]
            fontname = range[2]
        elif not doing_sans_conversion:
            uniindex = 1
            fontname = mpl.rcParams['mathtext.default']
    if fontname in ('rm', 'it'):
        uniindex = stix_glyph_fixes.get(uniindex, uniindex)
    if fontname in ('it', 'rm', 'bf', 'bfit') and 57344 <= uniindex <= 63743:
        fontname = 'nonuni' + fontname
    return (fontname, uniindex)