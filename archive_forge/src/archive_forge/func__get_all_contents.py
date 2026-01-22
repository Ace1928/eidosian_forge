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
@staticmethod
def _get_all_contents(page):
    page = mupdf.pdf_page_from_fz_page(page.this)
    res = JM_read_contents(page.obj())
    result = JM_BinFromBuffer(res)
    return result