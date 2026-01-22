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
class TruetypeFonts(Fonts, metaclass=abc.ABCMeta):
    """
    A generic base class for all font setups that use Truetype fonts
    (through FT2Font).
    """

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        super().__init__(default_font_prop, load_glyph_flags)
        self._get_info = functools.cache(self._get_info)
        self._fonts = {}
        self.fontmap: dict[str | int, str] = {}
        filename = findfont(self.default_font_prop)
        default_font = get_font(filename)
        self._fonts['default'] = default_font
        self._fonts['regular'] = default_font

    def _get_font(self, font: str | int) -> FT2Font:
        if font in self.fontmap:
            basename = self.fontmap[font]
        else:
            basename = T.cast(str, font)
        cached_font = self._fonts.get(basename)
        if cached_font is None and os.path.exists(basename):
            cached_font = get_font(basename)
            self._fonts[basename] = cached_font
            self._fonts[cached_font.postscript_name] = cached_font
            self._fonts[cached_font.postscript_name.lower()] = cached_font
        return T.cast(FT2Font, cached_font)

    def _get_offset(self, font: FT2Font, glyph: Glyph, fontsize: float, dpi: float) -> float:
        if font.postscript_name == 'Cmex10':
            return glyph.height / 64 / 2 + fontsize / 3 * dpi / 72
        return 0.0

    def _get_glyph(self, fontname: str, font_class: str, sym: str) -> tuple[FT2Font, int, bool]:
        raise NotImplementedError

    def _get_info(self, fontname: str, font_class: str, sym: str, fontsize: float, dpi: float) -> FontInfo:
        font, num, slanted = self._get_glyph(fontname, font_class, sym)
        font.set_size(fontsize, dpi)
        glyph = font.load_char(num, flags=self.load_glyph_flags)
        xmin, ymin, xmax, ymax = [val / 64.0 for val in glyph.bbox]
        offset = self._get_offset(font, glyph, fontsize, dpi)
        metrics = FontMetrics(advance=glyph.linearHoriAdvance / 65536.0, height=glyph.height / 64.0, width=glyph.width / 64.0, xmin=xmin, xmax=xmax, ymin=ymin + offset, ymax=ymax + offset, iceberg=glyph.horiBearingY / 64.0 + offset, slanted=slanted)
        return FontInfo(font=font, fontsize=fontsize, postscript_name=font.postscript_name, metrics=metrics, num=num, glyph=glyph, offset=offset)

    def get_xheight(self, fontname: str, fontsize: float, dpi: float) -> float:
        font = self._get_font(fontname)
        font.set_size(fontsize, dpi)
        pclt = font.get_sfnt_table('pclt')
        if pclt is None:
            metrics = self.get_metrics(fontname, mpl.rcParams['mathtext.default'], 'x', fontsize, dpi)
            return metrics.iceberg
        xHeight = pclt['xHeight'] / 64.0 * (fontsize / 12.0) * (dpi / 100.0)
        return xHeight

    def get_underline_thickness(self, font: str, fontsize: float, dpi: float) -> float:
        return 0.75 / 12.0 * fontsize * dpi / 72.0

    def get_kern(self, font1: str, fontclass1: str, sym1: str, fontsize1: float, font2: str, fontclass2: str, sym2: str, fontsize2: float, dpi: float) -> float:
        if font1 == font2 and fontsize1 == fontsize2:
            info1 = self._get_info(font1, fontclass1, sym1, fontsize1, dpi)
            info2 = self._get_info(font2, fontclass2, sym2, fontsize2, dpi)
            font = info1.font
            return font.get_kerning(info1.num, info2.num, KERNING_DEFAULT) / 64
        return super().get_kern(font1, fontclass1, sym1, fontsize1, font2, fontclass2, sym2, fontsize2, dpi)