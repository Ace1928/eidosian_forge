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
def _insert_font(self, fontfile=None, fontbuffer=None):
    """
        Utility: insert font from file or binary.
        """
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    if not fontfile and (not fontbuffer):
        raise ValueError(MSG_FILE_OR_BUFFER)
    value = JM_insert_font(pdf, None, fontfile, fontbuffer, 0, 0, 0, 0, 0, -1)
    return value