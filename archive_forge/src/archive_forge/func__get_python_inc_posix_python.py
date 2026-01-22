import os
import re
import sys
import sysconfig
import pathlib
from .errors import DistutilsPlatformError
from . import py39compat
from ._functools import pass_none
def _get_python_inc_posix_python(plat_specific):
    """
    Assume the executable is in the build directory. The
    pyconfig.h file should be in the same directory. Since
    the build directory may not be the source directory,
    use "srcdir" from the makefile to find the "Include"
    directory.
    """
    if not python_build:
        return
    if plat_specific:
        return _sys_home or project_base
    incdir = os.path.join(get_config_var('srcdir'), 'Include')
    return os.path.normpath(incdir)