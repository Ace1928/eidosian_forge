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
def _add_file_annot(self, point, buffer_, filename, ufilename=None, desc=None, icon=None):
    page = self._pdf_page()
    uf = ufilename if ufilename else filename
    d = desc if desc else filename
    p = JM_point_from_py(point)
    ASSERT_PDF(page)
    filebuf = JM_BufferFromBytes(buffer_)
    if not filebuf.m_internal:
        raise TypeError(MSG_BAD_BUFFER)
    annot = mupdf.pdf_create_annot(page, mupdf.PDF_ANNOT_FILE_ATTACHMENT)
    r = mupdf.pdf_annot_rect(annot)
    r = mupdf.fz_make_rect(p.x, p.y, p.x + r.x1 - r.x0, p.y + r.y1 - r.y0)
    mupdf.pdf_set_annot_rect(annot, r)
    flags = mupdf.PDF_ANNOT_IS_PRINT
    mupdf.pdf_set_annot_flags(annot, flags)
    if icon:
        mupdf.pdf_set_annot_icon_name(annot, icon)
    val = JM_embed_file(page.doc(), filebuf, filename, uf, d, 1)
    mupdf.pdf_dict_put(mupdf.pdf_annot_obj(annot), PDF_NAME('FS'), val)
    mupdf.pdf_dict_put_text_string(mupdf.pdf_annot_obj(annot), PDF_NAME('Contents'), filename)
    mupdf.pdf_update_annot(annot)
    mupdf.pdf_set_annot_rect(annot, r)
    mupdf.pdf_set_annot_flags(annot, flags)
    JM_add_annot_id(annot, 'A')
    return Annot(annot)