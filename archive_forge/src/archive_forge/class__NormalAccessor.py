import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import time
from collections import Sequence
from contextlib import contextmanager
from errno import EINVAL, ENOENT
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
class _NormalAccessor(_Accessor):

    def _wrap_strfunc(strfunc):

        @functools.wraps(strfunc)
        def wrapped(pathobj, *args):
            return strfunc(str(pathobj), *args)
        return staticmethod(wrapped)

    def _wrap_binary_strfunc(strfunc):

        @functools.wraps(strfunc)
        def wrapped(pathobjA, pathobjB, *args):
            return strfunc(str(pathobjA), str(pathobjB), *args)
        return staticmethod(wrapped)
    stat = _wrap_strfunc(os.stat)
    lstat = _wrap_strfunc(os.lstat)
    open = _wrap_strfunc(os.open)
    listdir = _wrap_strfunc(os.listdir)
    chmod = _wrap_strfunc(os.chmod)
    if hasattr(os, 'lchmod'):
        lchmod = _wrap_strfunc(os.lchmod)
    else:

        def lchmod(self, pathobj, mode):
            raise NotImplementedError('lchmod() not available on this system')
    mkdir = _wrap_strfunc(os.mkdir)
    unlink = _wrap_strfunc(os.unlink)
    rmdir = _wrap_strfunc(os.rmdir)
    rename = _wrap_binary_strfunc(os.rename)
    if sys.version_info >= (3, 3):
        replace = _wrap_binary_strfunc(os.replace)
    if nt:
        if supports_symlinks:
            symlink = _wrap_binary_strfunc(os.symlink)
        else:

            def symlink(a, b, target_is_directory):
                raise NotImplementedError('symlink() not available on this system')
    else:

        @staticmethod
        def symlink(a, b, target_is_directory):
            return os.symlink(str(a), str(b))
    utime = _wrap_strfunc(os.utime)

    def readlink(self, path):
        return os.readlink(path)