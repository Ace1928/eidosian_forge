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
def CCompiler_show_customization(self):
    """
    Print the compiler customizations to stdout.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Printing is only done if the distutils log threshold is < 2.

    """
    try:
        self.get_version()
    except Exception:
        pass
    if log._global_log.threshold < 2:
        print('*' * 80)
        print(self.__class__)
        print(_compiler_to_string(self))
        print('*' * 80)