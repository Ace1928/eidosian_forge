from socket import (
from typing import (
from zope.interface import implementer
from twisted.internet._idna import _idnaBytes
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.internet.threads import deferToThreadPool
from twisted.logger import Logger
from twisted.python.compat import nativeString
@implementer(IResolverSimple)
class ComplexResolverSimplifier:
    """
    A converter from L{IHostnameResolver} to L{IResolverSimple}
    """

    def __init__(self, nameResolver: IHostnameResolver):
        """
        Create a L{ComplexResolverSimplifier} with an L{IHostnameResolver}.

        @param nameResolver: The L{IHostnameResolver} to use.
        """
        self._nameResolver = nameResolver

    def getHostByName(self, name: str, timeouts: Sequence[int]=()) -> 'Deferred[str]':
        """
        See L{IResolverSimple.getHostByName}

        @param name: see L{IResolverSimple.getHostByName}

        @param timeouts: see L{IResolverSimple.getHostByName}

        @return: see L{IResolverSimple.getHostByName}
        """
        result: 'Deferred[str]' = Deferred()
        self._nameResolver.resolveHostName(FirstOneWins(result), name, 0, [IPv4Address])
        return result