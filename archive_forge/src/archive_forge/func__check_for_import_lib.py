import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
def _check_for_import_lib():
    """Check if an import library for the Python runtime already exists."""
    major_version, minor_version = tuple(sys.version_info[:2])
    patterns = ['libpython%d%d.a', 'libpython%d%d.dll.a', 'libpython%d.%d.dll.a']
    stems = [sys.prefix]
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    elif hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix:
        stems.append(sys.real_prefix)
    sub_dirs = ['libs', 'lib']
    candidates = []
    for pat in patterns:
        filename = pat % (major_version, minor_version)
        for stem_dir in stems:
            for folder in sub_dirs:
                candidates.append(os.path.join(stem_dir, folder, filename))
    for fullname in candidates:
        if os.path.isfile(fullname):
            return (True, fullname)
    return (False, candidates[0])