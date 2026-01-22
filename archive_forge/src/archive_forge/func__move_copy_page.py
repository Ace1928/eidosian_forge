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
def _move_copy_page(self, pno, nb, before, copy):
    """Move or copy a PDF page reference."""
    pdf = _as_pdf_document(self)
    same = 0
    ASSERT_PDF(pdf)
    page1, parent1, i1 = pdf_lookup_page_loc(pdf, pno)
    kids1 = mupdf.pdf_dict_get(parent1, PDF_NAME('Kids'))
    page2, parent2, i2 = pdf_lookup_page_loc(pdf, nb)
    kids2 = mupdf.pdf_dict_get(parent2, PDF_NAME('Kids'))
    if before:
        pos = i2
    else:
        pos = i2 + 1
    same = mupdf.pdf_objcmp(kids1, kids2)
    if not copy and same != 0:
        mupdf.pdf_dict_put(page1, PDF_NAME('Parent'), parent2)
    mupdf.pdf_array_insert(kids2, page1, pos)
    if same != 0:
        parent = parent2
        while parent.m_internal:
            count = mupdf.pdf_dict_get_int(parent, PDF_NAME('Count'))
            mupdf.pdf_dict_put_int(parent, PDF_NAME('Count'), count + 1)
            parent = mupdf.pdf_dict_get(parent, PDF_NAME('Parent'))
        if not copy:
            mupdf.pdf_array_delete(kids1, i1)
            parent = parent1
            while parent.m_internal:
                count = mupdf.pdf_dict_get_int(parent, PDF_NAME('Count'))
                mupdf.pdf_dict_put_int(parent, PDF_NAME('Count'), count - 1)
                parent = mupdf.pdf_dict_get(parent, PDF_NAME('Parent'))
    elif copy:
        parent = parent2
        while parent.m_internal:
            count = mupdf.pdf_dict_get_int(parent, PDF_NAME('Count'))
            mupdf.pdf_dict_put_int(parent, PDF_NAME('Count'), count + 1)
            parent = mupdf.pdf_dict_get(parent, PDF_NAME('Parent'))
    elif i1 < pos:
        mupdf.pdf_array_delete(kids1, i1)
    else:
        mupdf.pdf_array_delete(kids1, i1 + 1)
    if pdf.m_internal.rev_page_map:
        mupdf.ll_pdf_drop_page_tree(pdf.m_internal)
    self._reset_page_refs()