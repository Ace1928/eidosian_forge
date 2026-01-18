import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
def version_match(self, version_string):
    v = self.gnu_version_match(version_string)
    if not v or v[0] != 'gfortran':
        return None
    v = v[1]
    if LooseVersion(v) >= '4':
        pass
    elif sys.platform == 'win32':
        for key in ['version_cmd', 'compiler_f77', 'compiler_f90', 'compiler_fix', 'linker_so', 'linker_exe']:
            self.executables[key].append('-mno-cygwin')
    return v