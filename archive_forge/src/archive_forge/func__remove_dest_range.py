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
def _remove_dest_range(pdf, numbers):
    pagecount = mupdf.pdf_count_pages(pdf)
    for i in range(pagecount):
        n1 = i
        if n1 in numbers:
            continue
        pageref = mupdf.pdf_lookup_page_obj(pdf, i)
        annots = mupdf.pdf_dict_get(pageref, PDF_NAME('Annots'))
        if not annots.m_internal:
            continue
        len_ = mupdf.pdf_array_len(annots)
        for j in range(len_ - 1, -1, -1):
            o = mupdf.pdf_array_get(annots, j)
            if not mupdf.pdf_name_eq(mupdf.pdf_dict_get(o, PDF_NAME('Subtype')), PDF_NAME('Link')):
                continue
            action = mupdf.pdf_dict_get(o, PDF_NAME('A'))
            dest = mupdf.pdf_dict_get(o, PDF_NAME('Dest'))
            if action.m_internal:
                if not mupdf.pdf_name_eq(mupdf.pdf_dict_get(action, PDF_NAME('S')), PDF_NAME('GoTo')):
                    continue
                dest = mupdf.pdf_dict_get(action, PDF_NAME('D'))
            pno = -1
            if mupdf.pdf_is_array(dest):
                target = mupdf.pdf_array_get(dest, 0)
                pno = mupdf.pdf_lookup_page_number(pdf, target)
            elif mupdf.pdf_is_string(dest):
                location, _, _ = mupdf.fz_resolve_link(pdf.super(), mupdf.pdf_to_text_string(dest))
                pno = location.page
            if pno < 0:
                continue
            n1 = pno
            if n1 in numbers:
                mupdf.pdf_array_delete(annots, j)