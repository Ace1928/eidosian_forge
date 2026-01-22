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
def JM_get_annot_by_xref(page, xref):
    """
    retrieve annot by its xref
    """
    assert isinstance(page, mupdf.PdfPage)
    found = 0
    annot = mupdf.pdf_first_annot(page)
    while 1:
        if not annot.m_internal:
            break
        if xref == mupdf.pdf_to_num(mupdf.pdf_annot_obj(annot)):
            found = 1
            break
        annot = mupdf.pdf_next_annot(annot)
    if not found:
        raise Exception('xref %d is not an annot of this page' % xref)
    return annot