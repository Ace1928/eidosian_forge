import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def write_all_str(module_file, module_all_list):
    """Write the proper __all__ based on available operators."""
    module_file.write(os.linesep)
    module_file.write(os.linesep)
    all_str = '__all__ = [' + ', '.join(["'%s'" % s for s in module_all_list]) + ']'
    module_file.write(all_str)