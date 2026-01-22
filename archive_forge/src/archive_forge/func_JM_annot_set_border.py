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
def JM_annot_set_border(border, doc, annot_obj):
    assert isinstance(border, dict)
    obj = None
    dashlen = 0
    nwidth = border.get(dictkey_width)
    ndashes = border.get(dictkey_dashes)
    nstyle = border.get(dictkey_style)
    nclouds = border.get('clouds')
    oborder = JM_annot_border(annot_obj)
    mupdf.pdf_dict_del(annot_obj, PDF_NAME('BS'))
    mupdf.pdf_dict_del(annot_obj, PDF_NAME('BE'))
    mupdf.pdf_dict_del(annot_obj, PDF_NAME('Border'))
    if nwidth < 0:
        nwidth = oborder.get(dictkey_width)
    if ndashes is None:
        ndashes = oborder.get(dictkey_dashes)
    if nstyle is None:
        nstyle = oborder.get(dictkey_style)
    if nclouds < 0:
        nclouds = oborder.get('clouds')
    if isinstance(ndashes, tuple) and len(ndashes) > 0:
        dashlen = len(ndashes)
        darr = mupdf.pdf_new_array(doc, dashlen)
        for d in ndashes:
            mupdf.pdf_array_push_int(darr, d)
        mupdf.pdf_dict_putl(annot_obj, darr, PDF_NAME('BS'), PDF_NAME('D'))
    mupdf.pdf_dict_putl(annot_obj, mupdf.pdf_new_real(nwidth), PDF_NAME('BS'), PDF_NAME('W'))
    if dashlen == 0:
        obj = JM_get_border_style(nstyle)
    else:
        obj = PDF_NAME('D')
    mupdf.pdf_dict_putl(annot_obj, obj, PDF_NAME('BS'), PDF_NAME('S'))
    if nclouds > 0:
        mupdf.pdf_dict_put_dict(annot_obj, PDF_NAME('BE'), 2)
        obj = mupdf.pdf_dict_get(annot_obj, PDF_NAME('BE'))
        mupdf.pdf_dict_put(obj, PDF_NAME('S'), PDF_NAME('C'))
        mupdf.pdf_dict_put_int(obj, PDF_NAME('I'), nclouds)