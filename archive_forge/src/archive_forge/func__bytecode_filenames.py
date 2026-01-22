import os
import importlib.util
import sys
from distutils.core import Command
from distutils.errors import DistutilsOptionError
def _bytecode_filenames(self, py_filenames):
    bytecode_files = []
    for py_file in py_filenames:
        ext = os.path.splitext(os.path.normcase(py_file))[1]
        if ext != PYTHON_SOURCE_EXTENSION:
            continue
        if self.compile:
            bytecode_files.append(importlib.util.cache_from_source(py_file, optimization=''))
        if self.optimize > 0:
            bytecode_files.append(importlib.util.cache_from_source(py_file, optimization=self.optimize))
    return bytecode_files