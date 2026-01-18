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
def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
    """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.

        Parameters
        ----------
        objects : list
            List of object files to include.
        output_dir : str
            Output directory to place generated object files.
        extra_dll_dir : str
            Output directory to place extra DLL files that need to be
            included on Windows.

        Returns
        -------
        converted_objects : list of str
             List of converted object files.
             Note that the number of output files is not necessarily
             the same as inputs.

        """
    raise NotImplementedError()