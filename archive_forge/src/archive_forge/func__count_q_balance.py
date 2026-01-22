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
def _count_q_balance(self):
    """Count missing graphic state pushs and pops.

        Returns:
            A pair of integers (push, pop). Push is the number of missing
            PDF "q" commands, pop is the number of "Q" commands.
            A balanced graphics state for the page will be reached if its
            /Contents is prepended with 'push' copies of string "q
"
            and appended with 'pop' copies of "
Q".
        """
    page = _as_pdf_page(self)
    res = mupdf.pdf_dict_get(page.obj(), mupdf.PDF_ENUM_NAME_Resources)
    cont = mupdf.pdf_dict_get(page.obj(), mupdf.PDF_ENUM_NAME_Contents)
    pdf = _as_pdf_document(self.parent)
    return mupdf.pdf_count_q_balance_outparams_fn(pdf, res, cont)