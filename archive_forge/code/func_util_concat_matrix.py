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
def util_concat_matrix(m1, m2):
    return JM_py_from_matrix(mupdf.fz_concat(JM_matrix_from_py(m1), JM_matrix_from_py(m2)))