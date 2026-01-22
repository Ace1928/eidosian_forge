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
def _getAP(self):
    if g_use_extra:
        assert isinstance(self.this, mupdf.PdfAnnot)
        ret = extra.Annot_getAP(self.this)
        assert isinstance(ret, bytes)
        return ret
    else:
        r = None
        res = None
        annot = self.this
        assert isinstance(annot, mupdf.PdfAnnot)
        annot_obj = mupdf.pdf_annot_obj(annot)
        ap = mupdf.pdf_dict_getl(annot_obj, PDF_NAME('AP'), PDF_NAME('N'))
        if mupdf.pdf_is_stream(ap):
            res = mupdf.pdf_load_stream(ap)
        if res and res.m_internal:
            r = JM_BinFromBuffer(res)
        return r