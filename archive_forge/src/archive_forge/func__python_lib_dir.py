import contextlib
import os
import re
import sys
from distutils._log import log
from site import USER_BASE
from .._modified import newer_group
from ..core import Command
from ..errors import (
from ..extension import Extension
from ..sysconfig import customize_compiler, get_config_h_filename, get_python_version
from ..util import get_platform
@staticmethod
def _python_lib_dir(sysconfig):
    """
        Resolve Python's library directory for building extensions
        that rely on a shared Python library.

        See python/cpython#44264 and python/cpython#48686
        """
    if not sysconfig.get_config_var('Py_ENABLE_SHARED'):
        return
    if sysconfig.python_build:
        yield '.'
        return
    if sys.platform == 'zos':
        installed_dir = sysconfig.get_config_var('base')
        lib_dir = sysconfig.get_config_var('platlibdir')
        yield os.path.join(installed_dir, lib_dir)
    else:
        yield sysconfig.get_config_var('LIBDIR')