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
def JM_annot_colors(annot_obj):
    res = dict()
    bc = list()
    fc = list()
    o = mupdf.pdf_dict_get(annot_obj, mupdf.PDF_ENUM_NAME_C)
    if mupdf.pdf_is_array(o):
        n = mupdf.pdf_array_len(o)
        for i in range(n):
            col = mupdf.pdf_to_real(mupdf.pdf_array_get(o, i))
            bc.append(col)
    res[dictkey_stroke] = bc
    o = mupdf.pdf_dict_gets(annot_obj, 'IC')
    if mupdf.pdf_is_array(o):
        n = mupdf.pdf_array_len(o)
        for i in range(n):
            col = mupdf.pdf_to_real(mupdf.pdf_array_get(o, i))
            fc.append(col)
    res[dictkey_fill] = fc
    return res