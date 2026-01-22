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
def JM_font_descender(font):
    """
    need own versions of ascender / descender
    """
    assert isinstance(font, mupdf.FzFont)
    if g_skip_quad_corrections:
        return -0.2
    ret = mupdf.fz_font_descender(font)
    return ret