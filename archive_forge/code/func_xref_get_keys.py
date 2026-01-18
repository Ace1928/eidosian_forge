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
def xref_get_keys(self, xref):
    """Get the keys of PDF dict object at 'xref'. Use -1 for the PDF trailer."""
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    xreflen = mupdf.pdf_xref_len(pdf)
    if not _INRANGE(xref, 1, xreflen - 1) and xref != -1:
        raise ValueError(MSG_BAD_XREF)
    if xref > 0:
        obj = mupdf.pdf_load_object(pdf, xref)
    else:
        obj = mupdf.pdf_trailer(pdf)
    n = mupdf.pdf_dict_len(obj)
    rc = []
    if n == 0:
        return rc
    for i in range(n):
        key = mupdf.pdf_to_name(mupdf.pdf_dict_get_key(obj, i))
        rc.append(key)
    return rc