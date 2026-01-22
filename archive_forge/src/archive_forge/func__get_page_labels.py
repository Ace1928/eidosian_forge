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
def _get_page_labels(self):
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    rc = []
    pagelabels = mupdf.pdf_new_name('PageLabels')
    obj = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), pagelabels)
    if not obj.m_internal:
        return rc
    nums = mupdf.pdf_resolve_indirect(mupdf.pdf_dict_get(obj, PDF_NAME('Nums')))
    if nums.m_internal:
        JM_get_page_labels(rc, nums)
        return rc
    nums = mupdf.pdf_resolve_indirect(mupdf.pdf_dict_getl(obj, PDF_NAME('Kids'), PDF_NAME('Nums')))
    if nums.m_internal:
        JM_get_page_labels(rc, nums)
        return rc
    kids = mupdf.pdf_resolve_indirect(mupdf.pdf_dict_get(obj, PDF_NAME('Kids')))
    if not kids.m_internal or not mupdf.pdf_is_array(kids):
        return rc
    n = mupdf.pdf_array_len(kids)
    for i in range(n):
        nums = mupdf.pdf_resolve_indirect(mupdf.pdf_dict_get(mupdf.pdf_array_get(kids, i)), PDF_NAME('Nums'))
        JM_get_page_labels(rc, nums)
    return rc