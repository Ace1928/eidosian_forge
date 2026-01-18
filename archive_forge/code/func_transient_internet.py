from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
@contextlib.contextmanager
def transient_internet(resource_name, timeout=30.0, errnos=()):
    """Return a context manager that raises ResourceDenied when various issues
    with the Internet connection manifest themselves as exceptions."""
    default_errnos = [('ECONNREFUSED', 111), ('ECONNRESET', 104), ('EHOSTUNREACH', 113), ('ENETUNREACH', 101), ('ETIMEDOUT', 110)]
    default_gai_errnos = [('EAI_AGAIN', -3), ('EAI_FAIL', -4), ('EAI_NONAME', -2), ('EAI_NODATA', -5), ('WSANO_DATA', 11004)]
    denied = ResourceDenied('Resource %r is not available' % resource_name)
    captured_errnos = errnos
    gai_errnos = []
    if not captured_errnos:
        captured_errnos = [getattr(errno, name, num) for name, num in default_errnos]
        gai_errnos = [getattr(socket, name, num) for name, num in default_gai_errnos]

    def filter_error(err):
        n = getattr(err, 'errno', None)
        if isinstance(err, socket.timeout) or (isinstance(err, socket.gaierror) and n in gai_errnos) or n in captured_errnos:
            if not verbose:
                sys.stderr.write(denied.args[0] + '\n')
            exc = denied
            exc.__cause__ = err
            raise exc
    old_timeout = socket.getdefaulttimeout()
    try:
        if timeout is not None:
            socket.setdefaulttimeout(timeout)
        yield
    except IOError as err:
        while True:
            a = err.args
            if len(a) >= 1 and isinstance(a[0], IOError):
                err = a[0]
            elif len(a) >= 2 and isinstance(a[1], IOError):
                err = a[1]
            else:
                break
        filter_error(err)
        raise
    finally:
        socket.setdefaulttimeout(old_timeout)