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
def JM_get_border_style(style):
    """
    return pdf_obj "border style" from Python str
    """
    val = mupdf.PDF_ENUM_NAME_S
    if style is None:
        return val
    s = style
    if s.startswith('b') or s.startswith('B'):
        val = mupdf.PDF_ENUM_NAME_B
    elif s.startswith('d') or s.startswith('D'):
        val = mupdf.PDF_ENUM_NAME_D
    elif s.startswith('i') or s.startswith('I'):
        val = mupdf.PDF_ENUM_NAME_I
    elif s.startswith('u') or s.startswith('U'):
        val = mupdf.PDF_ENUM_NAME_U
    elif s.startswith('s') or s.startswith('S'):
        val = mupdf.PDF_ENUM_NAME_S
    return val