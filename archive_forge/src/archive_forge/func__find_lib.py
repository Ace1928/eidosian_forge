import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def _find_lib(self, lib_dir, lib, exts):
    assert is_string(lib_dir)
    if sys.platform == 'win32':
        lib_prefixes = ['', 'lib']
    else:
        lib_prefixes = ['lib']
    for ext in exts:
        for prefix in lib_prefixes:
            p = self.combine_paths(lib_dir, prefix + lib + ext)
            if p:
                break
        if p:
            assert len(p) == 1
            if ext == '.dll.a':
                lib += '.dll'
            if ext == '.lib':
                lib = prefix + lib
            return lib
    return False