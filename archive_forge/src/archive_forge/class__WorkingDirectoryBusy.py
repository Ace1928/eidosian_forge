from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
class _WorkingDirectoryBusy(Exception):
    """
    A working directory was specified to the runner, but another test run is
    currently using that directory.
    """