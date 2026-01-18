from __future__ import annotations
from typing_extensions import Protocol
from twisted.internet.defer import gatherResults
from twisted.names.dns import (
from twisted.names.hosts import Resolver, searchFileFor, searchFileForAll
from twisted.python.filepath import FilePath
from twisted.trial.unittest import SynchronousTestCase

        Like L{test_lookupAddressMalformed}, but for
        L{hosts.Resolver.lookupIPV6Address}.
        