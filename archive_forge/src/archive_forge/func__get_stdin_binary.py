from __future__ import absolute_import, print_function, division
import os
import io
import gzip
import sys
import bz2
import zipfile
from contextlib import contextmanager
import subprocess
import logging
from petl.errors import ArgumentError
from petl.compat import urlopen, StringIO, BytesIO, string_types, PY2
def _get_stdin_binary():
    try:
        return sys.stdin.buffer
    except AttributeError:
        pass
    try:
        fd = sys.stdin.fileno()
        return os.fdopen(fd, 'rb', 0)
    except Exception:
        pass
    try:
        return sys.__stdin__.buffer
    except AttributeError:
        pass
    try:
        fd = sys.__stdin__.fileno()
        return os.fdopen(fd, 'rb', 0)
    except Exception:
        pass
    return sys.stdin