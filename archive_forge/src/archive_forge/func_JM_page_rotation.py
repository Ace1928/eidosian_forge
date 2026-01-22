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
def JM_page_rotation(page):
    """
    return a PDF page's /Rotate value: one of (0, 90, 180, 270)
    """
    rotate = 0
    obj = mupdf.pdf_dict_get_inheritable(page.obj(), mupdf.PDF_ENUM_NAME_Rotate)
    rotate = mupdf.pdf_to_int(obj)
    rotate = JM_norm_rotation(rotate)
    return rotate