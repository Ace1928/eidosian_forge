from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def _read_win_env_flag(name: str, default: bool) -> bool:
    """Read a boolean flag from an environment variable on Windows.

    :return:
        On Windows, the flag, or the ``default`` value if absent or ambiguous.
        On all other operating systems, ``False``.

    :note: This only accesses the environment on Windows.
    """
    if os.name != 'nt':
        return False
    try:
        value = os.environ[name]
    except KeyError:
        return default
    _logger.warning('The %s environment variable is deprecated. Its effect has never been documented and changes without warning.', name)
    adjusted_value = value.strip().lower()
    if adjusted_value in {'', '0', 'false', 'no'}:
        return False
    if adjusted_value in {'1', 'true', 'yes'}:
        return True
    _logger.warning('%s has unrecognized value %r, treating as %r.', name, value, default)
    return default