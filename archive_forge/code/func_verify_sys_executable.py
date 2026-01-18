from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def verify_sys_executable(path: str) -> t.Optional[str]:
    """Verify that the given path references the current Python interpreter. If not, return the expected path, otherwise return None."""
    if path == sys.executable:
        return None
    if os.path.realpath(path) == os.path.realpath(sys.executable):
        return None
    expected_executable = raw_command([path, '-c', 'import sys; print(sys.executable)'], capture=True)[0]
    if expected_executable == sys.executable:
        return None
    return expected_executable