import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def _get_directory_containing_module(self, module_name):
    module = sys.modules[module_name]
    full_path = os.path.abspath(module.__file__)
    if os.path.basename(full_path).lower().startswith('__init__.py'):
        return os.path.dirname(os.path.dirname(full_path))
    else:
        return os.path.dirname(full_path)