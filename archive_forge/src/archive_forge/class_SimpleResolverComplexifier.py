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
@implementer(IHostnameResolver)
class SimpleResolverComplexifier:
    """
    A converter from L{IResolverSimple} to L{IHostnameResolver}.
    """
    _log = Logger()

    def __init__(self, simpleResolver: IResolverSimple):
        """
        Construct a L{SimpleResolverComplexifier} with an L{IResolverSimple}.
        """
        self._simpleResolver = simpleResolver

    def resolveHostName(self, resolutionReceiver: IResolutionReceiver, hostName: str, portNumber: int=0, addressTypes: Optional[Sequence[Type[IAddress]]]=None, transportSemantics: str='TCP') -> IHostResolution:
        """
        See L{IHostnameResolver.resolveHostName}

        @param resolutionReceiver: see interface

        @param hostName: see interface

        @param portNumber: see interface

        @param addressTypes: see interface

        @param transportSemantics: see interface

        @return: see interface
        """
        try:
            hostName_bytes = hostName.encode('ascii')
        except UnicodeEncodeError:
            hostName_bytes = _idnaBytes(hostName)
        hostName = nativeString(hostName_bytes)
        resolution = HostResolution(hostName)
        resolutionReceiver.resolutionBegan(resolution)
        self._simpleResolver.getHostByName(hostName).addCallback(lambda address: resolutionReceiver.addressResolved(IPv4Address('TCP', address, portNumber))).addErrback(lambda error: None if error.check(DNSLookupError) else self._log.failure('while looking up {name} with {resolver}', error, name=hostName, resolver=self._simpleResolver)).addCallback(lambda nothing: resolutionReceiver.resolutionComplete())
        return resolution