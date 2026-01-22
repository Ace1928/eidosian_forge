import os
import sys
import re
from pathlib import Path
from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
from distutils.util import split_quoted, strtobool
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils
from .environment import EnvironmentConfig
def _environment_hook(self, name, hook_name):
    if hook_name is None:
        return None
    if is_string(hook_name):
        if hook_name.startswith('self.'):
            hook_name = hook_name[5:]
            hook = getattr(self, hook_name)
            return hook()
        elif hook_name.startswith('exe.'):
            hook_name = hook_name[4:]
            var = self.executables[hook_name]
            if var:
                return var[0]
            else:
                return None
        elif hook_name.startswith('flags.'):
            hook_name = hook_name[6:]
            hook = getattr(self, 'get_flags_' + hook_name)
            return hook()
    else:
        return hook_name()