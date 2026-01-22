import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def addsitepackages(known_paths, prefixes=None):
    """Add site-packages to sys.path"""
    _trace('Processing global site-packages')
    for sitedir in getsitepackages(prefixes):
        if os.path.isdir(sitedir):
            addsitedir(sitedir, known_paths)
    return known_paths