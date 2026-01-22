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
def JM_read_contents(pageref):
    """
    Read and concatenate a PDF page's /Conents object(s) in a buffer
    """
    assert isinstance(pageref, mupdf.PdfObj), f'{type(pageref)}'
    contents = mupdf.pdf_dict_get(pageref, mupdf.PDF_ENUM_NAME_Contents)
    if mupdf.pdf_is_array(contents):
        res = mupdf.FzBuffer(1024)
        for i in range(mupdf.pdf_array_len(contents)):
            if i > 0:
                mupdf.fz_append_byte(res, 32)
            obj = mupdf.pdf_array_get(contents, i)
            if mupdf.pdf_is_stream(obj):
                nres = mupdf.pdf_load_stream(obj)
                mupdf.fz_append_buffer(res, nres)
    elif contents.m_internal:
        res = mupdf.pdf_load_stream(contents)
    return res