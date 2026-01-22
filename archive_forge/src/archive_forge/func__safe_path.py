from distutils.filelist import FileList as _FileList
from distutils.errors import DistutilsInternalError
from distutils.util import convert_path
from distutils import log
import distutils.errors
import distutils.filelist
import functools
import os
import re
import sys
import time
import collections
from .._importlib import metadata
from .. import _entry_points, _normalization
from . import _requirestxt
from setuptools import Command
from setuptools.command.sdist import sdist
from setuptools.command.sdist import walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.command import bdist_egg
import setuptools.unicode_utils as unicode_utils
from setuptools.glob import glob
from setuptools.extern import packaging
from ..warnings import SetuptoolsDeprecationWarning
def _safe_path(self, path):
    enc_warn = "'%s' not %s encodable -- skipping"
    u_path = unicode_utils.filesys_decode(path)
    if u_path is None:
        log.warn("'%s' in unexpected encoding -- skipping" % path)
        return False
    utf8_path = unicode_utils.try_encode(u_path, 'utf-8')
    if utf8_path is None:
        log.warn(enc_warn, path, 'utf-8')
        return False
    try:
        is_egg_info = '.egg-info' in u_path or b'.egg-info' in utf8_path
        if self.ignore_egg_info_dir and is_egg_info:
            return False
        if os.path.exists(u_path) or os.path.exists(utf8_path):
            return True
    except UnicodeEncodeError:
        log.warn(enc_warn, path, sys.getfilesystemencoding())