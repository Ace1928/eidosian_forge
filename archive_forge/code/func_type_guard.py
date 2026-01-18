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
def type_guard(sequence: c.Sequence[t.Any], guard_type: t.Type[C]) -> t.TypeGuard[c.Sequence[C]]:
    """
    Raises an exception if any item in the given sequence does not match the specified guard type.
    Use with assert so that type checkers are aware of the type guard.
    """
    invalid_types = set((type(item) for item in sequence if not isinstance(item, guard_type)))
    if not invalid_types:
        return True
    invalid_type_names = sorted((str(item) for item in invalid_types))
    raise Exception(f'Sequence required to contain only {guard_type} includes: {', '.join(invalid_type_names)}')