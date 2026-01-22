import os
import re
import sys
import sysconfig
import pathlib
from .errors import DistutilsPlatformError
from . import py39compat
from ._functools import pass_none
def _is_parent(dir_a, dir_b):
    """
    Return True if a is a parent of b.
    """
    return os.path.normcase(dir_a).startswith(os.path.normcase(dir_b))