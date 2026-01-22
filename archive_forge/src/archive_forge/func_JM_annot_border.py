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
def JM_annot_border(annot_obj):
    dash_py = list()
    style = None
    width = -1
    clouds = -1
    obj = None
    obj = mupdf.pdf_dict_get(annot_obj, PDF_NAME('Border'))
    if mupdf.pdf_is_array(obj):
        width = mupdf.pdf_to_real(mupdf.pdf_array_get(obj, 2))
        if mupdf.pdf_array_len(obj) == 4:
            dash = mupdf.pdf_array_get(obj, 3)
            for i in range(mupdf.pdf_array_len(dash)):
                val = mupdf.pdf_to_int(mupdf.pdf_array_get(dash, i))
                dash_py.append(val)
    bs_o = mupdf.pdf_dict_get(annot_obj, PDF_NAME('BS'))
    if bs_o.m_internal:
        width = mupdf.pdf_to_real(mupdf.pdf_dict_get(bs_o, PDF_NAME('W')))
        style = mupdf.pdf_to_name(mupdf.pdf_dict_get(bs_o, PDF_NAME('S')))
        if style == '':
            style = None
        obj = mupdf.pdf_dict_get(bs_o, PDF_NAME('D'))
        if obj.m_internal:
            for i in range(mupdf.pdf_array_len(obj)):
                val = mupdf.pdf_to_int(mupdf.pdf_array_get(obj, i))
                dash_py.append(val)
    obj = mupdf.pdf_dict_get(annot_obj, PDF_NAME('BE'))
    if obj.m_internal:
        clouds = mupdf.pdf_to_int(mupdf.pdf_dict_get(obj, PDF_NAME('I')))
    res = dict()
    res[dictkey_width] = width
    res[dictkey_dashes] = tuple(dash_py)
    res[dictkey_style] = style
    res['clouds'] = clouds
    return res