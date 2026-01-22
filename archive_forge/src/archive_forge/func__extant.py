import os
import re
import sys
import sysconfig
import pathlib
from .errors import DistutilsPlatformError
from . import py39compat
from ._functools import pass_none
@pass_none
def _extant(path):
    """
    Replace path with None if it doesn't exist.
    """
    return path if os.path.exists(path) else None