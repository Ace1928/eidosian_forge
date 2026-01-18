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
@property
def transformation_matrix(self):
    """Page transformation matrix."""
    CheckParent(self)
    ctm = mupdf.FzMatrix()
    page = self._pdf_page()
    if not page.m_internal:
        return JM_py_from_matrix(ctm)
    mediabox = mupdf.FzRect(mupdf.FzRect.Fixed_UNIT)
    mupdf.pdf_page_transform(page, mediabox, ctm)
    val = JM_py_from_matrix(ctm)
    if self.rotation % 360 == 0:
        val = Matrix(val)
    else:
        val = Matrix(1, 0, 0, -1, 0, self.cropbox.height)
    return val