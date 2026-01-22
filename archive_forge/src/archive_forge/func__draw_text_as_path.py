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
def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath, mtext=None):
    writer = self.writer
    writer.comment(s)
    glyph_map = self._glyph_map
    text2path = self._text2path
    color = rgb2hex(gc.get_rgb())
    fontsize = prop.get_size_in_points()
    style = {}
    if color != '#000000':
        style['fill'] = color
    alpha = gc.get_alpha() if gc.get_forced_alpha() else gc.get_rgb()[3]
    if alpha != 1:
        style['opacity'] = _short_float_fmt(alpha)
    font_scale = fontsize / text2path.FONT_SCALE
    attrib = {'style': _generate_css(style), 'transform': _generate_transform([('translate', (x, y)), ('rotate', (-angle,)), ('scale', (font_scale, -font_scale))])}
    writer.start('g', attrib=attrib)
    if not ismath:
        font = text2path._get_font(prop)
        _glyphs = text2path.get_glyphs_with_font(font, s, glyph_map=glyph_map, return_new_glyphs_only=True)
        glyph_info, glyph_map_new, rects = _glyphs
        self._update_glyph_map_defs(glyph_map_new)
        for glyph_id, xposition, yposition, scale in glyph_info:
            attrib = {'xlink:href': f'#{glyph_id}'}
            if xposition != 0.0:
                attrib['x'] = _short_float_fmt(xposition)
            if yposition != 0.0:
                attrib['y'] = _short_float_fmt(yposition)
            writer.element('use', attrib=attrib)
    else:
        if ismath == 'TeX':
            _glyphs = text2path.get_glyphs_tex(prop, s, glyph_map=glyph_map, return_new_glyphs_only=True)
        else:
            _glyphs = text2path.get_glyphs_mathtext(prop, s, glyph_map=glyph_map, return_new_glyphs_only=True)
        glyph_info, glyph_map_new, rects = _glyphs
        self._update_glyph_map_defs(glyph_map_new)
        for char_id, xposition, yposition, scale in glyph_info:
            char_id = self._adjust_char_id(char_id)
            writer.element('use', transform=_generate_transform([('translate', (xposition, yposition)), ('scale', (scale,))]), attrib={'xlink:href': f'#{char_id}'})
        for verts, codes in rects:
            path = Path(verts, codes)
            path_data = self._convert_path(path, simplify=False)
            writer.element('path', d=path_data)
    writer.end('g')