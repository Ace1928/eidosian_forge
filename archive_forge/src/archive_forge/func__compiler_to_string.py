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
def _compiler_to_string(compiler):
    props = []
    mx = 0
    keys = list(compiler.executables.keys())
    for key in ['version', 'libraries', 'library_dirs', 'object_switch', 'compile_switch', 'include_dirs', 'define', 'undef', 'rpath', 'link_objects']:
        if key not in keys:
            keys.append(key)
    for key in keys:
        if hasattr(compiler, key):
            v = getattr(compiler, key)
            mx = max(mx, len(key))
            props.append((key, repr(v)))
    fmt = '%-' + repr(mx + 1) + 's = %s'
    lines = [fmt % prop for prop in props]
    return '\n'.join(lines)