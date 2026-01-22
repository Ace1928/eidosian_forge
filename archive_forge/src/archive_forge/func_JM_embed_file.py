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
def JM_embed_file(pdf, buf, filename, ufilename, desc, compress):
    """
    embed a new file in a PDF (not only /EmbeddedFiles entries)
    """
    len_ = 0
    val = mupdf.pdf_new_dict(pdf, 6)
    mupdf.pdf_dict_put_dict(val, PDF_NAME('CI'), 4)
    ef = mupdf.pdf_dict_put_dict(val, PDF_NAME('EF'), 4)
    mupdf.pdf_dict_put_text_string(val, PDF_NAME('F'), filename)
    mupdf.pdf_dict_put_text_string(val, PDF_NAME('UF'), ufilename)
    mupdf.pdf_dict_put_text_string(val, PDF_NAME('Desc'), desc)
    mupdf.pdf_dict_put(val, PDF_NAME('Type'), PDF_NAME('Filespec'))
    bs = b'  '
    f = mupdf.pdf_add_stream(pdf, mupdf.fz_new_buffer_from_copied_data(bs), mupdf.PdfObj(), 0)
    mupdf.pdf_dict_put(ef, PDF_NAME('F'), f)
    JM_update_stream(pdf, f, buf, compress)
    len_, _ = mupdf.fz_buffer_storage(buf)
    mupdf.pdf_dict_put_int(f, PDF_NAME('DL'), len_)
    mupdf.pdf_dict_put_int(f, PDF_NAME('Length'), len_)
    params = mupdf.pdf_dict_put_dict(f, PDF_NAME('Params'), 4)
    mupdf.pdf_dict_put_int(params, PDF_NAME('Size'), len_)
    return val