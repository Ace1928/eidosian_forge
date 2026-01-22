import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
import shutil as _shutil
import stat as _stat
import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
def _dont_follow_symlinks(func, path, *args):
    if func in _os.supports_follow_symlinks:
        func(path, *args, follow_symlinks=False)
    elif _os.name == 'nt' or not _os.path.islink(path):
        func(path, *args)