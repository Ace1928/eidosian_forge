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
def CheckMorph(o: typing.Any) -> bool:
    if not bool(o):
        return False
    if not (type(o) in (list, tuple) and len(o) == 2):
        raise ValueError('morph must be a sequence of length 2')
    if not (len(o[0]) == 2 and len(o[1]) == 6):
        raise ValueError('invalid morph parm 0')
    if not o[1][4] == o[1][5] == 0:
        raise ValueError('invalid morph parm 1')
    return True