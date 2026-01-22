from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
class _NoTrialMarker(Exception):
    """
    No trial marker file could be found.

    Raised when trial attempts to remove a trial temporary working directory
    that does not contain a marker file.
    """