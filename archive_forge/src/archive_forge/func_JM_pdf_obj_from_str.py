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
def JM_pdf_obj_from_str(doc, src):
    """
    create PDF object from given string (new in v1.14.0: MuPDF dropped it)
    """
    buffer_ = mupdf.fz_new_buffer_from_copied_data(bytes(src, 'utf8'))
    stream = mupdf.fz_open_buffer(buffer_)
    lexbuf = mupdf.PdfLexbuf(mupdf.PDF_LEXBUF_SMALL)
    result = mupdf.pdf_parse_stm_obj(doc, stream, lexbuf)
    return result