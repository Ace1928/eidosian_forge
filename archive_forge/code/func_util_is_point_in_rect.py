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
def util_is_point_in_rect(p, r):
    return mupdf.fz_is_point_inside_rect(JM_point_from_py(p), JM_rect_from_py(r))