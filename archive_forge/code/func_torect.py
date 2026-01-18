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
def torect(self, r):
    """Return matrix that converts to target rect."""
    r = Rect(r)
    if self.is_infinite or self.is_empty or r.is_infinite or r.is_empty:
        raise ValueError('rectangles must be finite and not empty')
    return Matrix(1, 0, 0, 1, -self.x0, -self.y0) * Matrix(r.width / self.width, r.height / self.height) * Matrix(1, 0, 0, 1, r.x0, r.y0)