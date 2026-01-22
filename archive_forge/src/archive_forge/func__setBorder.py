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
def _setBorder(self, border, doc, xref):
    pdf = _as_pdf_document(doc)
    if not pdf:
        return
    link_obj = mupdf.pdf_new_indirect(pdf, xref, 0)
    if not link_obj.m_internal:
        return
    b = JM_annot_set_border(border, pdf, link_obj)
    return b