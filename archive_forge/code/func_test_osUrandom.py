from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
def test_osUrandom(self) -> None:
    """
        L{RandomFactory._osUrandom} should work as a random source whenever
        L{os.urandom} is available.
        """
    self._check(self.factory._osUrandom)