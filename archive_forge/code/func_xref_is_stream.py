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
def xref_is_stream(self, xref=0):
    """Check if xref is a stream object."""
    pdf = _as_pdf_document(self)
    if not pdf:
        return False
    return bool(mupdf.pdf_obj_num_is_stream(pdf, xref))