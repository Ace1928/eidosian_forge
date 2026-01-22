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
def JM_BinFromBuffer(buffer_):
    """
    Turn fz_buffer into a Python bytes object
    """
    assert isinstance(buffer_, mupdf.FzBuffer)
    ret = mupdf.fz_buffer_extract_copy(buffer_)
    return ret