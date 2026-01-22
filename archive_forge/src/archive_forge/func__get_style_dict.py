import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
def _get_style_dict(self, gc, rgbFace):
    """Generate a style string from the GraphicsContext and rgbFace."""
    attrib = {}
    forced_alpha = gc.get_forced_alpha()
    if gc.get_hatch() is not None:
        attrib['fill'] = f'url(#{self._get_hatch(gc, rgbFace)})'
        if rgbFace is not None and len(rgbFace) == 4 and (rgbFace[3] != 1.0) and (not forced_alpha):
            attrib['fill-opacity'] = _short_float_fmt(rgbFace[3])
    elif rgbFace is None:
        attrib['fill'] = 'none'
    else:
        if tuple(rgbFace[:3]) != (0, 0, 0):
            attrib['fill'] = rgb2hex(rgbFace)
        if len(rgbFace) == 4 and rgbFace[3] != 1.0 and (not forced_alpha):
            attrib['fill-opacity'] = _short_float_fmt(rgbFace[3])
    if forced_alpha and gc.get_alpha() != 1.0:
        attrib['opacity'] = _short_float_fmt(gc.get_alpha())
    offset, seq = gc.get_dashes()
    if seq is not None:
        attrib['stroke-dasharray'] = ','.join((_short_float_fmt(val) for val in seq))
        attrib['stroke-dashoffset'] = _short_float_fmt(float(offset))
    linewidth = gc.get_linewidth()
    if linewidth:
        rgb = gc.get_rgb()
        attrib['stroke'] = rgb2hex(rgb)
        if not forced_alpha and rgb[3] != 1.0:
            attrib['stroke-opacity'] = _short_float_fmt(rgb[3])
        if linewidth != 1.0:
            attrib['stroke-width'] = _short_float_fmt(linewidth)
        if gc.get_joinstyle() != 'round':
            attrib['stroke-linejoin'] = gc.get_joinstyle()
        if gc.get_capstyle() != 'butt':
            attrib['stroke-linecap'] = _capstyle_d[gc.get_capstyle()]
    return attrib