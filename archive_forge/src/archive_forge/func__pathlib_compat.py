import io
import posixpath
import zipfile
import itertools
import contextlib
import sys
import pathlib
def _pathlib_compat(path):
    """
    For path-like objects, convert to a filename for compatibility
    on Python 3.6.1 and earlier.
    """
    try:
        return path.__fspath__()
    except AttributeError:
        return str(path)