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
def _remove_toc_item(self, xref):
    pdf = _as_pdf_document(self)
    item = mupdf.pdf_new_indirect(pdf, xref, 0)
    mupdf.pdf_dict_del(item, PDF_NAME('Dest'))
    mupdf.pdf_dict_del(item, PDF_NAME('A'))
    color = mupdf.pdf_new_array(pdf, 3)
    for i in range(3):
        mupdf.pdf_array_push_real(color, 0.8)
    mupdf.pdf_dict_put(item, PDF_NAME('C'), color)