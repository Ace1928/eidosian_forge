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
def afmFontProperty(fontpath, font):
    """
    Extract information from an AFM font file.

    Parameters
    ----------
    fontpath : str
        The filename corresponding to *font*.
    font : AFM
        The AFM font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.
    """
    name = font.get_familyname()
    fontname = font.get_fontname().lower()
    if font.get_angle() != 0 or 'italic' in name.lower():
        style = 'italic'
    elif 'oblique' in name.lower():
        style = 'oblique'
    else:
        style = 'normal'
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'
    weight = font.get_weight().lower()
    if weight not in weight_dict:
        weight = 'normal'
    if 'demi cond' in fontname:
        stretch = 'semi-condensed'
    elif any((word in fontname for word in ['narrow', 'cond'])):
        stretch = 'condensed'
    elif any((word in fontname for word in ['wide', 'expanded', 'extended'])):
        stretch = 'expanded'
    else:
        stretch = 'normal'
    size = 'scalable'
    return FontEntry(fontpath, name, style, variant, weight, stretch, size)