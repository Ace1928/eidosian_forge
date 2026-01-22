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
def JM_get_font(fontname, fontfile, fontbuffer, script, lang, ordering, is_bold, is_italic, is_serif, embed):
    """
    return a fz_font from a number of parameters
    """

    def fertig(font):
        if not font.m_internal:
            raise RuntimeError(MSG_FONT_FAILED)
        if not font.m_internal.flags.never_embed:
            mupdf.fz_set_font_embedding(font, embed)
        return font
    index = 0
    font = None
    if fontfile:
        font = mupdf.fz_new_font_from_file(None, fontfile, index, 0)
        return fertig(font)
    if fontbuffer:
        res = JM_BufferFromBytes(fontbuffer)
        font = mupdf.fz_new_font_from_buffer(None, res, index, 0)
        return fertig(font)
    if ordering > -1:
        font = mupdf.fz_new_cjk_font(ordering)
        return fertig(font)
    if fontname:
        font = mupdf.fz_new_base14_font(fontname)
        if font.m_internal:
            return fertig(font)
        font = mupdf.fz_new_builtin_font(fontname, is_bold, is_italic)
        return fertig(font)
    data, size, index = mupdf.fz_lookup_noto_font(script, lang)
    font = None
    if data:
        font = mupdf.fz_new_font_from_memory(None, data, size, index, 0)
    if font.m_internal:
        return fertig(font)
    font = mupdf.fz_load_fallback_font(script, lang, is_serif, is_bold, is_italic)
    return fertig(font)