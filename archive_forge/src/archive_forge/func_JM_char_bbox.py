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
def JM_char_bbox(line, ch):
    """
    return rect of char quad
    """
    q = JM_char_quad(line, ch)
    r = mupdf.fz_rect_from_quad(q)
    if not line.m_internal.wmode:
        return r
    if r.y1 < r.y0 + ch.m_internal.size:
        r.y0 = r.y1 - ch.m_internal.size
    return r