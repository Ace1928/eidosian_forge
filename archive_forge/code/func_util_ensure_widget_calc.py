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
def util_ensure_widget_calc(annot):
    """
    Ensure that widgets with /AA/C JavaScript are in array AcroForm/CO
    """
    annot_obj = mupdf.pdf_annot_obj(annot.this)
    pdf = mupdf.pdf_get_bound_document(annot_obj)
    PDFNAME_CO = mupdf.pdf_new_name('CO')
    acro = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('AcroForm'))
    CO = mupdf.pdf_dict_get(acro, PDFNAME_CO)
    if not CO.this:
        CO = mupdf.pdf_dict_put_array(acro, PDFNAME_CO, 2)
    n = mupdf.pdf_array_len(CO)
    found = 0
    xref = mupdf.pdf_to_num(annot_obj)
    for i in range(n):
        nxref = mupdf.pdf_to_num(mupdf.pdf_array_get(CO, i))
        if xref == nxref:
            found = 1
            break
    if not found:
        mupdf.pdf_array_push(CO, mupdf.pdf_new_indirect(pdf, xref, 0))