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
@staticmethod
def _process_site_dirs(site_dirs):
    if site_dirs is None:
        return
    normpath = map(normalize_path, sys.path)
    site_dirs = [os.path.expanduser(s.strip()) for s in site_dirs.split(',')]
    for d in site_dirs:
        if not os.path.isdir(d):
            log.warn('%s (in --site-dirs) does not exist', d)
        elif normalize_path(d) not in normpath:
            raise DistutilsOptionError(d + ' (in --site-dirs) is not on sys.path')
        else:
            yield normalize_path(d)