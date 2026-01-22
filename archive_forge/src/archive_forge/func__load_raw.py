from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
def _load_raw(self):
    paths = []
    dirty = saw_import = False
    seen = dict.fromkeys(self.sitedirs)
    f = open(self.filename, 'rt', encoding=py39.LOCALE_ENCODING)
    for line in f:
        path = line.rstrip()
        paths.append(path)
        if line.startswith(('import ', 'from ')):
            saw_import = True
            continue
        stripped_path = path.strip()
        if not stripped_path or stripped_path.startswith('#'):
            continue
        normalized_path = normalize_path(os.path.join(self.basedir, path))
        if normalized_path in seen or not os.path.exists(normalized_path):
            log.debug('cleaned up dirty or duplicated %r', path)
            dirty = True
            paths.pop()
            continue
        seen[normalized_path] = 1
    f.close()
    while paths and (not paths[-1].strip()):
        paths.pop()
        dirty = True
    return (paths, dirty or (paths and saw_import))