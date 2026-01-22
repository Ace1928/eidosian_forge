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
def JM_UnicodeFromStr(s):
    if s is None:
        return ''
    if isinstance(s, bytes):
        s = s.decode('utf8')
    assert isinstance(s, str), f'type(s)={type(s)!r} s={s!r}'
    return s