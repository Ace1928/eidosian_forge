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
def _as_fz_document(document):
    """
    Returns document as a mupdf.FzDocument, upcasting as required. Raises
    'document closed' exception if closed.
    """
    if isinstance(document, Document):
        if document.is_closed:
            raise ValueError('document closed')
        document = document.this
    if isinstance(document, mupdf.FzDocument):
        return document
    elif isinstance(document, mupdf.PdfDocument):
        return document.super()
    elif document is None:
        assert 0, f'document is None'
    else:
        assert 0, f'Unrecognised type(document)={type(document)!r}'