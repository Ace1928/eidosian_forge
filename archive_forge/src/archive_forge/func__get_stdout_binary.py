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
def _get_stdout_binary():
    try:
        return sys.stdout.buffer
    except AttributeError:
        pass
    try:
        fd = sys.stdout.fileno()
        return os.fdopen(fd, 'ab', 0)
    except Exception:
        pass
    try:
        return sys.__stdout__.buffer
    except AttributeError:
        pass
    try:
        fd = sys.__stdout__.fileno()
        return os.fdopen(fd, 'ab', 0)
    except Exception:
        pass
    return sys.stdout