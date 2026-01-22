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
def _insertFont(self, fontname, bfname, fontfile, fontbuffer, set_simple, idx, wmode, serif, encoding, ordering):
    page = self._pdf_page()
    ASSERT_PDF(page)
    pdf = page.doc()
    value = JM_insert_font(pdf, bfname, fontfile, fontbuffer, set_simple, idx, wmode, serif, encoding, ordering)
    resources = mupdf.pdf_dict_get_inheritable(page.obj(), PDF_NAME('Resources'))
    fonts = mupdf.pdf_dict_get(resources, PDF_NAME('Font'))
    if not fonts.m_internal:
        fonts = mupdf.pdf_new_dict(pdf, 5)
        mupdf.pdf_dict_putl(page.obj(), fonts, PDF_NAME('Resources'), PDF_NAME('Font'))
    _, xref = JM_INT_ITEM(value, 0)
    if not xref:
        raise RuntimeError('cannot insert font')
    font_obj = mupdf.pdf_new_indirect(pdf, xref, 0)
    mupdf.pdf_dict_puts(fonts, fontname, font_obj)
    return value