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
def _add_text_annot(self, point, text, icon=None):
    page = self._pdf_page()
    p = JM_point_from_py(point)
    ASSERT_PDF(page)
    annot = mupdf.pdf_create_annot(page, mupdf.PDF_ANNOT_TEXT)
    r = mupdf.pdf_annot_rect(annot)
    r = mupdf.fz_make_rect(p.x, p.y, p.x + r.x1 - r.x0, p.y + r.y1 - r.y0)
    mupdf.pdf_set_annot_rect(annot, r)
    mupdf.pdf_set_annot_contents(annot, text)
    if icon:
        mupdf.pdf_set_annot_icon_name(annot, icon)
    mupdf.pdf_update_annot(annot)
    JM_add_annot_id(annot, 'A')
    return Annot(annot)