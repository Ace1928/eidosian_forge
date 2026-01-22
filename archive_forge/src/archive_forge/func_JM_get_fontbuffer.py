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
def JM_get_fontbuffer(doc, xref):
    """
    Return the contents of a font file, identified by xref
    """
    if xref < 1:
        return
    o = mupdf.pdf_load_object(doc, xref)
    desft = mupdf.pdf_dict_get(o, PDF_NAME('DescendantFonts'))
    if desft.m_internal:
        obj = mupdf.pdf_resolve_indirect(mupdf.pdf_array_get(desft, 0))
        obj = mupdf.pdf_dict_get(obj, PDF_NAME('FontDescriptor'))
    else:
        obj = mupdf.pdf_dict_get(o, PDF_NAME('FontDescriptor'))
    if not obj.m_internal:
        message(f'invalid font - FontDescriptor missing')
        return
    o = obj
    stream = None
    obj = mupdf.pdf_dict_get(o, PDF_NAME('FontFile'))
    if obj.m_internal:
        stream = obj
    obj = mupdf.pdf_dict_get(o, PDF_NAME('FontFile2'))
    if obj.m_internal:
        stream = obj
    obj = mupdf.pdf_dict_get(o, PDF_NAME('FontFile3'))
    if obj.m_internal:
        stream = obj
        obj = mupdf.pdf_dict_get(obj, PDF_NAME('Subtype'))
        if obj.m_internal and (not mupdf.pdf_is_name(obj)):
            message('invalid font descriptor subtype')
            return
        if mupdf.pdf_name_eq(obj, PDF_NAME('Type1C')):
            pass
        elif mupdf.pdf_name_eq(obj, PDF_NAME('CIDFontType0C')):
            pass
        elif mupdf.pdf_name_eq(obj, PDF_NAME('OpenType')):
            pass
        else:
            message('warning: unhandled font type {pdf_to_name(ctx, obj)!r}')
    if not stream:
        message('warning: unhandled font type')
        return
    return mupdf.pdf_load_stream(stream)