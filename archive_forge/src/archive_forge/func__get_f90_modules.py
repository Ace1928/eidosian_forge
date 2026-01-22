import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def _get_f90_modules(source):
    """Return a list of Fortran f90 module names that
    given source file defines.
    """
    if not f90_ext_match(source):
        return []
    modules = []
    with open(source) as f:
        for line in f:
            m = f90_module_name_match(line)
            if m:
                name = m.group('name')
                modules.append(name)
    return modules