import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def until_no_eintr(f, *a, **kw):
    """Run f(*a, **kw), retrying if an EINTR error occurs.

    WARNING: you must be certain that it is safe to retry the call repeatedly
    if EINTR does occur.  This is typically only true for low-level operations
    like os.read.  If in any doubt, don't use this.

    Keep in mind that this is not a complete solution to EINTR.  There is
    probably code in the Python standard library and other dependencies that
    may encounter EINTR if a signal arrives (and there is signal handler for
    that signal).  So this function can reduce the impact for IO that breezy
    directly controls, but it is not a complete solution.
    """
    while True:
        try:
            return f(*a, **kw)
        except OSError as e:
            if e.errno == errno.EINTR:
                continue
            raise