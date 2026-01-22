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
def _getPDFfileid(self):
    """Get PDF file id."""
    pdf = _as_pdf_document(self)
    if not pdf:
        return
    idlist = []
    identity = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('ID'))
    if identity.m_internal:
        n = mupdf.pdf_array_len(identity)
        for i in range(n):
            o = mupdf.pdf_array_get(identity, i)
            text = mupdf.pdf_to_text_string(o)
            hex_ = binascii.hexlify(text)
            idlist.append(hex_)
    return idlist