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
def as_requirement(self):
    """Return a ``Requirement`` that matches this distribution exactly"""
    if isinstance(self.parsed_version, packaging.version.Version):
        spec = '%s==%s' % (self.project_name, self.parsed_version)
    else:
        spec = '%s===%s' % (self.project_name, self.parsed_version)
    return Requirement.parse(spec)