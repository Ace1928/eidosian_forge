import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from sysconfig import get_config_var, get_config_vars, get_path
from .runners import (
from .util import (
def any_fortran_src(srcs):
    return _any_X(srcs, FortranCompilerRunner)