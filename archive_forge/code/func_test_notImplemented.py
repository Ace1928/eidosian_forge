from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase
def test_notImplemented(self) -> None:
    """
        L{hosts.Resolver} fails with L{NotImplementedError} for L{IResolver}
        methods it doesn't implement.
        """
    self.failureResultOf(self.resolver.lookupMailExchange(b'EXAMPLE'), NotImplementedError)