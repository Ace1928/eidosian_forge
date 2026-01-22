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
def _add_freetext_annot(self, rect, text, fontsize=11, fontname=None, text_color=None, fill_color=None, border_color=None, align=0, rotate=0):
    page = self._pdf_page()
    nfcol, fcol = JM_color_FromSequence(fill_color)
    ntcol, tcol = JM_color_FromSequence(text_color)
    r = JM_rect_from_py(rect)
    if mupdf.fz_is_infinite_rect(r) or mupdf.fz_is_empty_rect(r):
        raise ValueError(MSG_BAD_RECT)
    annot = mupdf.pdf_create_annot(page, mupdf.PDF_ANNOT_FREE_TEXT)
    annot_obj = mupdf.pdf_annot_obj(annot)
    mupdf.pdf_set_annot_contents(annot, text)
    mupdf.pdf_set_annot_rect(annot, r)
    mupdf.pdf_dict_put_int(annot_obj, PDF_NAME('Rotate'), rotate)
    mupdf.pdf_dict_put_int(annot_obj, PDF_NAME('Q'), align)
    if nfcol > 0:
        mupdf.pdf_set_annot_color(annot, fcol[:nfcol])
    JM_make_annot_DA(annot, ntcol, tcol, fontname, fontsize)
    mupdf.pdf_update_annot(annot)
    JM_add_annot_id(annot, 'A')
    val = Annot(annot)
    ap = val._getAP()
    BT = ap.find(b'BT')
    ET = ap.rfind(b'ET') + 2
    ap = ap[BT:ET]
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    if rotate in (90, -90, 270):
        w, h = (h, w)
    re = b'0 0 %g %g re' % (w, h)
    ap = re + b'\nW\nn\n' + ap
    ope = None
    bwidth = b''
    fill_string = ColorCode(fill_color, 'f').encode()
    if fill_string:
        fill_string += b'\n'
        ope = b'f'
    stroke_string = ColorCode(border_color, 'c').encode()
    if stroke_string:
        stroke_string += b'\n'
        bwidth = b'1 w\n'
        ope = b'S'
    if fill_string and stroke_string:
        ope = b'B'
    if ope is not None:
        ap = bwidth + fill_string + stroke_string + re + b'\n' + ope + b'\n' + ap
    val._setAP(ap)
    return val