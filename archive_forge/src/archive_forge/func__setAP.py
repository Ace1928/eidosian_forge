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
def _setAP(self, buffer_, rect=0):
    try:
        annot = self.this
        annot_obj = mupdf.pdf_annot_obj(annot)
        page = mupdf.pdf_annot_page(annot)
        apobj = mupdf.pdf_dict_getl(annot_obj, PDF_NAME('AP'), PDF_NAME('N'))
        if not apobj.m_internal:
            raise RuntimeError(MSG_BAD_APN)
        if not mupdf.pdf_is_stream(apobj):
            raise RuntimeError(MSG_BAD_APN)
        res = JM_BufferFromBytes(buffer_)
        if not res.m_internal:
            raise ValueError(MSG_BAD_BUFFER)
        JM_update_stream(page.doc(), apobj, res, 1)
        if rect:
            bbox = mupdf.pdf_dict_get_rect(annot_obj, PDF_NAME('Rect'))
            mupdf.pdf_dict_put_rect(apobj, PDF_NAME('BBox'), bbox)
    except Exception:
        if g_exceptions_verbose:
            exception_info()