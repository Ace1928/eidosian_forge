import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def JM_char_quad(line, ch):
    """
    re-compute char quad if ascender/descender values make no sense
    """
    if 1 and g_use_extra:
        return mupdf.FzQuad(extra.JM_char_quad(line.m_internal, ch.m_internal))
    assert isinstance(line, mupdf.FzStextLine)
    assert isinstance(ch, mupdf.FzStextChar)
    if g_skip_quad_corrections:
        return ch.quad
    if line.m_internal.wmode:
        return ch.quad
    font = mupdf.FzFont(mupdf.ll_fz_keep_font(ch.m_internal.font))
    asc = JM_font_ascender(font)
    dsc = JM_font_descender(font)
    fsize = ch.m_internal.size
    asc_dsc = asc - dsc + FLT_EPSILON
    if asc_dsc >= 1 and g_small_glyph_heights == 0:
        return mupdf.FzQuad(ch.m_internal.quad)
    fsize = ch.m_internal.size
    bbox = mupdf.fz_font_bbox(font)
    fwidth = bbox.x1 - bbox.x0
    if asc < 0.001:
        dsc = -0.1
        asc = 0.9
        asc_dsc = 1.0
    if g_small_glyph_heights or asc_dsc < 1:
        dsc = dsc / asc_dsc
        asc = asc / asc_dsc
    asc_dsc = asc - dsc
    asc = asc * fsize / asc_dsc
    dsc = dsc * fsize / asc_dsc
    c = line.m_internal.dir.x
    s = line.m_internal.dir.y
    trm1 = mupdf.fz_make_matrix(c, -s, s, c, 0, 0)
    trm2 = mupdf.fz_make_matrix(c, s, -s, c, 0, 0)
    if c == -1:
        trm1.d = 1
        trm2.d = 1
    xlate1 = mupdf.fz_make_matrix(1, 0, 0, 1, -ch.m_internal.origin.x, -ch.m_internal.origin.y)
    xlate2 = mupdf.fz_make_matrix(1, 0, 0, 1, ch.m_internal.origin.x, ch.m_internal.origin.y)
    quad = mupdf.fz_transform_quad(mupdf.FzQuad(ch.m_internal.quad), xlate1)
    quad = mupdf.fz_transform_quad(quad, trm1)
    if c == 1 and quad.ul.y > 0:
        quad.ul.y = asc
        quad.ur.y = asc
        quad.ll.y = dsc
        quad.lr.y = dsc
    else:
        quad.ul.y = -asc
        quad.ur.y = -asc
        quad.ll.y = -dsc
        quad.lr.y = -dsc
    if quad.ll.x < 0:
        quad.ll.x = 0
        quad.ul.x = 0
    cwidth = quad.lr.x - quad.ll.x
    if cwidth < FLT_EPSILON:
        glyph = mupdf.fz_encode_character(font, ch.m_internal.c)
        if glyph:
            fwidth = mupdf.fz_advance_glyph(font, glyph, line.m_internal.wmode)
            quad.lr.x = quad.ll.x + fwidth * fsize
            quad.ur.x = quad.lr.x
    quad = mupdf.fz_transform_quad(quad, trm2)
    quad = mupdf.fz_transform_quad(quad, xlate2)
    return quad