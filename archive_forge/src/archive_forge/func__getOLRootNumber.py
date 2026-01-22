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
def _getOLRootNumber(self):
    """Get xref of Outline Root, create it if missing."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    olroot = mupdf.pdf_dict_get(root, PDF_NAME('Outlines'))
    if not olroot.m_internal:
        olroot = mupdf.pdf_new_dict(pdf, 4)
        mupdf.pdf_dict_put(olroot, PDF_NAME('Type'), PDF_NAME('Outlines'))
        ind_obj = mupdf.pdf_add_object(pdf, olroot)
        mupdf.pdf_dict_put(root, PDF_NAME('Outlines'), ind_obj)
        olroot = mupdf.pdf_dict_get(root, PDF_NAME('Outlines'))
    return mupdf.pdf_to_num(olroot)