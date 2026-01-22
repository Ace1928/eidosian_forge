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
def _set_page_labels(self, labels):
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    pagelabels = mupdf.pdf_new_name('PageLabels')
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    mupdf.pdf_dict_del(root, pagelabels)
    mupdf.pdf_dict_putl(root, mupdf.pdf_new_array(pdf, 0), pagelabels, PDF_NAME('Nums'))
    xref = self.pdf_catalog()
    text = self.xref_object(xref, compressed=True)
    text = text.replace('/Nums[]', '/Nums[%s]' % labels)
    self.update_object(xref, text)