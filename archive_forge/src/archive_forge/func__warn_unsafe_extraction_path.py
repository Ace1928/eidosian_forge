import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
@staticmethod
def _warn_unsafe_extraction_path(path):
    """
        If the default extraction path is overridden and set to an insecure
        location, such as /tmp, it opens up an opportunity for an attacker to
        replace an extracted file with an unauthorized payload. Warn the user
        if a known insecure location is used.

        See Distribute #375 for more details.
        """
    if os.name == 'nt' and (not path.startswith(os.environ['windir'])):
        return
    mode = os.stat(path).st_mode
    if mode & stat.S_IWOTH or mode & stat.S_IWGRP:
        msg = 'Extraction path is writable by group/others and vulnerable to attack when used with get_resource_filename ({path}). Consider a more secure location (set with .set_extraction_path or the PYTHON_EGG_CACHE environment variable).'.format(**locals())
        warnings.warn(msg, UserWarning)