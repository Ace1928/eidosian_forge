import builtins
import inspect
import io
import keyword
import linecache
import os
import re
import sys
import sysconfig
import tokenize
import traceback
def _is_file_mine(self, file):
    filepath = os.path.abspath(file).lower()
    if not filepath.endswith('.py'):
        return False
    return not any((filepath.startswith(d) for d in self._lib_dirs))