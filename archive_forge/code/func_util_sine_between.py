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
def util_sine_between(C, P, Q):
    c = JM_point_from_py(C)
    p = JM_point_from_py(P)
    q = JM_point_from_py(Q)
    s = mupdf.fz_normalize_vector(mupdf.fz_make_point(q.x - p.x, q.y - p.y))
    m1 = mupdf.fz_make_matrix(1, 0, 0, 1, -p.x, -p.y)
    m2 = mupdf.fz_make_matrix(s.x, -s.y, s.y, s.x, 0, 0)
    m1 = mupdf.fz_concat(m1, m2)
    c = mupdf.fz_transform_point(c, m1)
    c = mupdf.fz_normalize_vector(c)
    return c.y