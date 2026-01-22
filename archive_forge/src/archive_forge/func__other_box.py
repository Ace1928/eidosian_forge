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
def _other_box(self, boxtype):
    rect = mupdf.FzRect(mupdf.FzRect.Fixed_INFINITE)
    page = mupdf.pdf_page_from_fz_page(self.this)
    if page.m_internal:
        obj = mupdf.pdf_dict_gets(page.obj(), boxtype)
        if mupdf.pdf_is_array(obj):
            rect = mupdf.pdf_to_rect(obj)
    if mupdf.fz_is_infinite_rect(rect):
        return
    return JM_py_from_rect(rect)