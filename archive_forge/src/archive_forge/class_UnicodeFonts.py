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
class UnicodeFonts(TruetypeFonts):
    """
    An abstract base class for handling Unicode fonts.

    While some reasonably complete Unicode fonts (such as DejaVu) may
    work in some situations, the only Unicode font I'm aware of with a
    complete set of math symbols is STIX.

    This class will "fallback" on the Bakoma fonts when a required
    symbol cannot be found in the font.
    """
    _cmr10_substitutions = {215: 163, 8722: 161}

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        fallback_rc = mpl.rcParams['mathtext.fallback']
        font_cls: type[TruetypeFonts] | None = {'stix': StixFonts, 'stixsans': StixSansFonts, 'cm': BakomaFonts}.get(fallback_rc)
        self._fallback_font = font_cls(default_font_prop, load_glyph_flags) if font_cls else None
        super().__init__(default_font_prop, load_glyph_flags)
        for texfont in 'cal rm tt it bf sf bfit'.split():
            prop = mpl.rcParams['mathtext.' + texfont]
            font = findfont(prop)
            self.fontmap[texfont] = font
        prop = FontProperties('cmex10')
        font = findfont(prop)
        self.fontmap['ex'] = font
        if isinstance(self._fallback_font, StixFonts):
            stixsizedaltfonts = {0: 'STIXGeneral', 1: 'STIXSizeOneSym', 2: 'STIXSizeTwoSym', 3: 'STIXSizeThreeSym', 4: 'STIXSizeFourSym', 5: 'STIXSizeFiveSym'}
            for size, name in stixsizedaltfonts.items():
                fullpath = findfont(name)
                self.fontmap[size] = fullpath
                self.fontmap[name] = fullpath
    _slanted_symbols = set('\\int \\oint'.split())

    def _map_virtual_font(self, fontname: str, font_class: str, uniindex: int) -> tuple[str, int]:
        return (fontname, uniindex)

    def _get_glyph(self, fontname: str, font_class: str, sym: str) -> tuple[FT2Font, int, bool]:
        try:
            uniindex = get_unicode_index(sym)
            found_symbol = True
        except ValueError:
            uniindex = ord('?')
            found_symbol = False
            _log.warning('No TeX to Unicode mapping for %a.', sym)
        fontname, uniindex = self._map_virtual_font(fontname, font_class, uniindex)
        new_fontname = fontname
        if found_symbol:
            if fontname == 'it' and uniindex < 65536:
                char = chr(uniindex)
                if unicodedata.category(char)[0] != 'L' or unicodedata.name(char).startswith('GREEK CAPITAL'):
                    new_fontname = 'rm'
            slanted = new_fontname == 'it' or sym in self._slanted_symbols
            found_symbol = False
            font = self._get_font(new_fontname)
            if font is not None:
                if uniindex in self._cmr10_substitutions and font.family_name == 'cmr10':
                    font = get_font(cbook._get_data_path('fonts/ttf/cmsy10.ttf'))
                    uniindex = self._cmr10_substitutions[uniindex]
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    found_symbol = True
        if not found_symbol:
            if self._fallback_font:
                if fontname in ('it', 'regular') and isinstance(self._fallback_font, StixFonts):
                    fontname = 'rm'
                g = self._fallback_font._get_glyph(fontname, font_class, sym)
                family = g[0].family_name
                if family in list(BakomaFonts._fontmap.values()):
                    family = 'Computer Modern'
                _log.info('Substituting symbol %s from %s', sym, family)
                return g
            else:
                if fontname in ('it', 'regular') and isinstance(self, StixFonts):
                    return self._get_glyph('rm', font_class, sym)
                _log.warning('Font %r does not have a glyph for %a [U+%x], substituting with a dummy symbol.', new_fontname, sym, uniindex)
                font = self._get_font('rm')
                uniindex = 164
                slanted = False
        return (font, uniindex, slanted)

    def get_sized_alternatives_for_symbol(self, fontname: str, sym: str) -> list[tuple[str, str]]:
        if self._fallback_font:
            return self._fallback_font.get_sized_alternatives_for_symbol(fontname, sym)
        return [(fontname, sym)]