from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
def ttfFontProperty(font):
    """
    Extract information from a TrueType font file.

    Parameters
    ----------
    font : `.FT2Font`
        The TrueType font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.

    """
    name = font.family_name
    sfnt = font.get_sfnt()
    mac_key = (1, 0, 0)
    ms_key = (3, 1, 1033)
    sfnt2 = sfnt.get((*mac_key, 2), b'').decode('latin-1').lower() or sfnt.get((*ms_key, 2), b'').decode('utf_16_be').lower()
    sfnt4 = sfnt.get((*mac_key, 4), b'').decode('latin-1').lower() or sfnt.get((*ms_key, 4), b'').decode('utf_16_be').lower()
    if sfnt4.find('oblique') >= 0:
        style = 'oblique'
    elif sfnt4.find('italic') >= 0:
        style = 'italic'
    elif sfnt2.find('regular') >= 0:
        style = 'normal'
    elif font.style_flags & ft2font.ITALIC:
        style = 'italic'
    else:
        style = 'normal'
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'
    wws_subfamily = 22
    typographic_subfamily = 16
    font_subfamily = 2
    styles = [sfnt.get((*mac_key, wws_subfamily), b'').decode('latin-1'), sfnt.get((*mac_key, typographic_subfamily), b'').decode('latin-1'), sfnt.get((*mac_key, font_subfamily), b'').decode('latin-1'), sfnt.get((*ms_key, wws_subfamily), b'').decode('utf-16-be'), sfnt.get((*ms_key, typographic_subfamily), b'').decode('utf-16-be'), sfnt.get((*ms_key, font_subfamily), b'').decode('utf-16-be')]
    styles = [*filter(None, styles)] or [font.style_name]

    def get_weight():
        os2 = font.get_sfnt_table('OS/2')
        if os2 and os2['version'] != 65535:
            return os2['usWeightClass']
        try:
            ps_font_info_weight = font.get_ps_font_info()['weight'].replace(' ', '') or ''
        except ValueError:
            pass
        else:
            for regex, weight in _weight_regexes:
                if re.fullmatch(regex, ps_font_info_weight, re.I):
                    return weight
        for style in styles:
            style = style.replace(' ', '')
            for regex, weight in _weight_regexes:
                if re.search(regex, style, re.I):
                    return weight
        if font.style_flags & ft2font.BOLD:
            return 700
        return 500
    weight = int(get_weight())
    if any((word in sfnt4 for word in ['narrow', 'condensed', 'cond'])):
        stretch = 'condensed'
    elif 'demi cond' in sfnt4:
        stretch = 'semi-condensed'
    elif any((word in sfnt4 for word in ['wide', 'expanded', 'extended'])):
        stretch = 'expanded'
    else:
        stretch = 'normal'
    if not font.scalable:
        raise NotImplementedError('Non-scalable fonts are not supported')
    size = 'scalable'
    return FontEntry(font.fname, name, style, variant, weight, stretch, size)