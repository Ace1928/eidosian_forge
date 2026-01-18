import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
def this_file_dir(stack_offset=1):
    """Returns the directory containing the module that calls this function."""
    return os.path.dirname(this_file(stack_offset=1 + stack_offset))