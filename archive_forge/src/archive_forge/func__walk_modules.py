import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _walk_modules(path):
    """Generate name and path of modules and packages on path."""
    for root, dirs, files in os.walk(path):
        files.sort()
        for f in files:
            if f[:2] != '__':
                if f.endswith(('.py', COMPILED_EXT)):
                    yield (f.rsplit('.', 1)[0], root)
        dirs.sort()
        for d in dirs:
            if d[:2] != '__':
                package_dir = osutils.pathjoin(root, d)
                fullpath = _get_package_init(package_dir)
                if fullpath is not None:
                    yield (d, package_dir)
        del dirs[:]