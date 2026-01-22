import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings
def _stringify_path(path):
    """
    Convert *path* to a string or unicode path if possible.
    """
    if isinstance(path, str):
        return os.path.expanduser(path)
    try:
        return os.path.expanduser(path.__fspath__())
    except AttributeError:
        pass
    raise TypeError('not a path-like object')