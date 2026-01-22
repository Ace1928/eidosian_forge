import os
import re
import sys
import sysconfig
import pathlib
from .errors import DistutilsPlatformError
from . import py39compat
from ._functools import pass_none
def _get_python_inc_nt(prefix, spec_prefix, plat_specific):
    if python_build:
        return os.path.join(prefix, 'include') + os.path.pathsep + os.path.join(prefix, 'PC')
    return os.path.join(prefix, 'include')