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
def util_point_in_quad(P, Q):
    p = JM_point_from_py(P)
    q = JM_quad_from_py(Q)
    return mupdf.fz_is_point_inside_quad(p, q)