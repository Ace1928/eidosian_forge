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
def _embfile_info(self, idx, infodict):
    pdf = _as_pdf_document(self)
    xref = 0
    ci_xref = 0
    trailer = mupdf.pdf_trailer(pdf)
    names = mupdf.pdf_dict_getl(trailer, PDF_NAME('Root'), PDF_NAME('Names'), PDF_NAME('EmbeddedFiles'), PDF_NAME('Names'))
    o = mupdf.pdf_array_get(names, 2 * idx + 1)
    ci = mupdf.pdf_dict_get(o, PDF_NAME('CI'))
    if ci.m_internal:
        ci_xref = mupdf.pdf_to_num(ci)
    infodict['collection'] = ci_xref
    name = mupdf.pdf_to_text_string(mupdf.pdf_dict_get(o, PDF_NAME('F')))
    infodict[dictkey_filename] = JM_EscapeStrFromStr(name)
    name = mupdf.pdf_to_text_string(mupdf.pdf_dict_get(o, PDF_NAME('UF')))
    infodict[dictkey_ufilename] = JM_EscapeStrFromStr(name)
    name = mupdf.pdf_to_text_string(mupdf.pdf_dict_get(o, PDF_NAME('Desc')))
    infodict[dictkey_desc] = JM_UnicodeFromStr(name)
    len_ = -1
    DL = -1
    fileentry = mupdf.pdf_dict_getl(o, PDF_NAME('EF'), PDF_NAME('F'))
    xref = mupdf.pdf_to_num(fileentry)
    o = mupdf.pdf_dict_get(fileentry, PDF_NAME('Length'))
    if o.m_internal:
        len_ = mupdf.pdf_to_int(o)
    o = mupdf.pdf_dict_get(fileentry, PDF_NAME('DL'))
    if o.m_internal:
        DL = mupdf.pdf_to_int(o)
    else:
        o = mupdf.pdf_dict_getl(fileentry, PDF_NAME('Params'), PDF_NAME('Size'))
        if o.m_internal:
            DL = mupdf.pdf_to_int(o)
    infodict[dictkey_size] = DL
    infodict[dictkey_length] = len_
    return xref