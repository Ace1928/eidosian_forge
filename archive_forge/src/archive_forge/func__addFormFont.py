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
def _addFormFont(self, name, font):
    """Add new form font."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    if not pdf:
        return
    fonts = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('AcroForm'), PDF_NAME('DR'), PDF_NAME('Font'))
    if not fonts.m_internal or not mupdf.pdf_is_dict(fonts):
        raise RuntimeError('PDF has no form fonts yet')
    k = mupdf.pdf_new_name(name)
    v = JM_pdf_obj_from_str(pdf, font)
    mupdf.pdf_dict_put(fonts, k, v)