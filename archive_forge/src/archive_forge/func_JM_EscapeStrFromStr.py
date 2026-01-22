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
def JM_EscapeStrFromStr(c):
    if c is None:
        return ''
    assert isinstance(c, str), f'type(c)={type(c)!r}'
    b = c.encode('utf8', 'surrogateescape')
    ret = ''
    for bb in b:
        ret += chr(bb)
    return ret