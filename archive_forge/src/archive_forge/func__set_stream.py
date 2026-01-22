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
def _set_stream(name, default):
    """
    Returns a stream to use based on environmental variable `name`.
    """
    t = os.environ.get(name)
    if t is None:
        return default
    elif t.startswith('fd:'):
        return open(int(t[3:]), mode='w', closefd=False)
    elif t.startswith('path:'):
        return open(t[5:], 'w')
    elif t.startswith('path+:'):
        return open(t[6:], 'a')
    else:
        raise Exception(f'Unrecognised stream specification for {name!r} should match `fd:<int>`, `path:<string>` or `path+:<string>`: {t!r}')