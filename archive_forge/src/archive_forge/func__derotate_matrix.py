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
def _derotate_matrix(page):
    if isinstance(page, mupdf.PdfPage):
        return JM_py_from_matrix(JM_derotate_page_matrix(page))
    else:
        return JM_py_from_matrix(mupdf.FzMatrix())