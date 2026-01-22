import os
import re
import sys
import sysconfig
import pathlib
from .errors import DistutilsPlatformError
from . import py39compat
from ._functools import pass_none
def _get_python_inc_posix(prefix, spec_prefix, plat_specific):
    if IS_PYPY and sys.version_info < (3, 8):
        return os.path.join(prefix, 'include')
    return _get_python_inc_posix_python(plat_specific) or _extant(_get_python_inc_from_config(plat_specific, spec_prefix)) or _get_python_inc_posix_prefix(prefix)