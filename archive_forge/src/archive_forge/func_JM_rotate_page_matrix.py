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
def JM_rotate_page_matrix(page):
    """
    calculate page rotation matrices
    """
    if not page.m_internal:
        return mupdf.FzMatrix()
    rotation = JM_page_rotation(page)
    if rotation == 0:
        return mupdf.FzMatrix()
    cb_size = JM_cropbox_size(page.obj())
    w = cb_size.x
    h = cb_size.y
    if rotation == 90:
        m = mupdf.fz_make_matrix(0, 1, -1, 0, h, 0)
    elif rotation == 180:
        m = mupdf.fz_make_matrix(-1, 0, 0, -1, w, h)
    else:
        m = mupdf.fz_make_matrix(0, -1, 1, 0, 0, w)
    return m