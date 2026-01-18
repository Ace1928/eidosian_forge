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
def util_make_irect(*args, p0=None, p1=None, x0=None, y0=None, x1=None, y1=None):
    a, b, c, d = util_make_rect(*args, p0=p0, p1=p1, x0=x0, y0=y0, x1=x1, y1=y1)

    def convert(x):
        ret = int(x)
        return ret
    a = convert(a)
    b = convert(b)
    c = convert(c)
    d = convert(d)
    return (a, b, c, d)