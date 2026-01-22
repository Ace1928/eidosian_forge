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
def JM_point_from_py(p):
    """
    PySequence to fz_point. Default: (FZ_MIN_INF_RECT, FZ_MIN_INF_RECT)
    """
    if isinstance(p, mupdf.FzPoint):
        return p
    if isinstance(p, Point):
        return mupdf.FzPoint(p.x, p.y)
    if g_use_extra:
        return extra.JM_point_from_py(p)
    p0 = mupdf.FzPoint(0, 0)
    x = JM_FLOAT_ITEM(p, 0)
    y = JM_FLOAT_ITEM(p, 1)
    if x is None or y is None:
        return p0
    x = max(x, FZ_MIN_INF_RECT)
    y = max(y, FZ_MIN_INF_RECT)
    x = min(x, FZ_MAX_INF_RECT)
    y = min(y, FZ_MAX_INF_RECT)
    return mupdf.FzPoint(x, y)