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
@property
def _bbox(self):
    val = JM_py_from_rect(mupdf.fz_bound_text(self.this, mupdf.FzStrokeState(None), mupdf.FzMatrix()))
    val = Rect(val)
    return val