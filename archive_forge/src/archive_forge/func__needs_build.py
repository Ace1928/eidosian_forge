import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def _needs_build(obj, cc_args, extra_postargs, pp_opts):
    """
    Check if an objects needs to be rebuild based on its dependencies

    Parameters
    ----------
    obj : str
        object file

    Returns
    -------
    bool
    """
    dep_file = obj + '.d'
    if not os.path.exists(dep_file):
        return True
    with open(dep_file) as f:
        lines = f.readlines()
    cmdline = _commandline_dep_string(cc_args, extra_postargs, pp_opts)
    last_cmdline = lines[-1]
    if last_cmdline != cmdline:
        return True
    contents = ''.join(lines[:-1])
    deps = [x for x in shlex.split(contents, posix=True) if x != '\n' and (not x.endswith(':'))]
    try:
        t_obj = os.stat(obj).st_mtime
        for f in deps:
            if os.stat(f).st_mtime > t_obj:
                return True
    except OSError:
        return True
    return False