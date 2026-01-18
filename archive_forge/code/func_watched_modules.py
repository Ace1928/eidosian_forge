import asyncio
import fnmatch
import logging
import os
import sys
import types
import warnings
from contextlib import contextmanager
from bokeh.application.handlers import CodeHandler
from ..util import fullpath
from .state import state
def watched_modules():
    files = list(_watched_files)
    module_paths = {}
    for module_name in _modules | _local_modules:
        if module_name not in sys.modules:
            continue
        module = sys.modules[module_name]
        if not isinstance(module, types.ModuleType):
            continue
        path = getattr(module, '__file__', None)
        if not path:
            continue
        if path.endswith(('.pyc', '.pyo')):
            path = path[:-1]
        path = os.path.abspath(os.path.realpath(path))
        module_paths[path] = module_name
        files.append(path)
    return (module_paths, files)