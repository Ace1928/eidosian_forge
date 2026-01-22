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
@classmethod
def _build_from_requirements(cls, req_spec):
    """
        Build a working set from a requirement spec. Rewrites sys.path.
        """
    ws = cls([])
    reqs = parse_requirements(req_spec)
    dists = ws.resolve(reqs, Environment())
    for dist in dists:
        ws.add(dist)
    for entry in sys.path:
        if entry not in ws.entries:
            ws.add_entry(entry)
    sys.path[:] = ws.entries
    return ws