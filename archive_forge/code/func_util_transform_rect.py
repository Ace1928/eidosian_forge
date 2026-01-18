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
def util_transform_rect(rect, matrix):
    if g_use_extra:
        return extra.util_transform_rect(rect, matrix)
    return JM_py_from_rect(mupdf.fz_transform_rect(JM_rect_from_py(rect), JM_matrix_from_py(matrix)))