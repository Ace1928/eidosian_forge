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
def PyUnicode_DecodeRawUnicodeEscape(s, errors='strict'):
    if not s:
        return ''
    if isinstance(s, str):
        rc = s.encode('utf8', errors=errors)
    elif isinstance(s, bytes):
        rc = s[:]
    ret = rc.decode('raw_unicode_escape', errors=errors)
    return ret