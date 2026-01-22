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
class EggProvider(NullProvider):
    """Provider based on a virtual filesystem"""

    def __init__(self, module):
        super().__init__(module)
        self._setup_prefix()

    def _setup_prefix(self):
        eggs = filter(_is_egg_path, _parents(self.module_path))
        egg = next(eggs, None)
        egg and self._set_egg(egg)

    def _set_egg(self, path):
        self.egg_name = os.path.basename(path)
        self.egg_info = os.path.join(path, 'EGG-INFO')
        self.egg_root = path