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
def _writeIMG(self, filename, format_, jpg_quality):
    pm = self.this
    if format_ == 1:
        mupdf.fz_save_pixmap_as_png(pm, filename)
    elif format_ == 2:
        mupdf.fz_save_pixmap_as_pnm(pm, filename)
    elif format_ == 3:
        mupdf.fz_save_pixmap_as_pam(pm, filename)
    elif format_ == 5:
        mupdf.fz_save_pixmap_as_psd(pm, filename)
    elif format_ == 6:
        mupdf.fz_save_pixmap_as_ps(pm, filename)
    elif format_ == 7:
        mupdf.fz_save_pixmap_as_jpeg(pm, filename, jpg_quality)
    else:
        mupdf.fz_save_pixmap_as_png(pm, filename)