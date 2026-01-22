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
def _deleteObject(self, xref):
    """Delete object."""
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    if not _INRANGE(xref, 1, mupdf.pdf_xref_len(pdf) - 1):
        raise ValueError(MSG_BAD_XREF)
    mupdf.pdf_delete_object(pdf, xref)