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
def verified_chmod(path: str, mode: int) -> None:
    """Perform chmod on the specified path and then verify the permissions were applied."""
    os.chmod(path, mode)
    executable = any((mode & perm for perm in (stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH)))
    if executable and (not os.access(path, os.X_OK)):
        raise ApplicationError(f'Path "{path}" should executable, but is not. Is the filesystem mounted with the "noexec" option?')