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
def JM_derotate_page_matrix(page):
    """
    just the inverse of rotation
    """
    mp = JM_rotate_page_matrix(page)
    return mupdf.fz_invert_matrix(mp)