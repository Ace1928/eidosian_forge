import functools
import os.path
from .errors import DistutilsFileError
from .py39compat import zip_strict
from ._functools import splat
def _newer(source, target):
    return not os.path.exists(target) or os.path.getmtime(source) > os.path.getmtime(target)