from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def _getReactor(self):
    """
        Get either the passed-in reactor or the global reactor.
        """
    if self.reactor is not None:
        reactor = self.reactor
    else:
        from twisted.internet import reactor
    return reactor