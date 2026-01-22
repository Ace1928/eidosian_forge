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
def JM_cropbox_size(page_obj):
    rect = JM_cropbox(page_obj)
    w = abs(rect.x1 - rect.x0)
    h = abs(rect.y1 - rect.y0)
    size = mupdf.fz_make_point(w, h)
    return size