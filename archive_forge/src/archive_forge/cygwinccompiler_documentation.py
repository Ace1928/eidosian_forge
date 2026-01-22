import os
import sys
import copy
from subprocess import Popen, PIPE, check_output
import re
from distutils.unixccompiler import UnixCCompiler
from distutils.file_util import write_file
from distutils.errors import (DistutilsExecError, CCompilerError,
from distutils.version import LooseVersion
from distutils.spawn import find_executable
Adds supports for rc and res files.