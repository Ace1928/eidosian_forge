from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IHostnameResolver(Interface):
    """
    An L{IHostnameResolver} can resolve a host name and port number into a
    series of L{IAddress} objects.

    @since: Twisted 17.1.0
    """

    def resolveHostName(resolutionReceiver: IResolutionReceiver, hostName: str, portNumber: int=0, addressTypes: Optional[Sequence[Type[IAddress]]]=None, transportSemantics: str='TCP') -> IHostResolution:
        """
        Initiate a hostname resolution.

        @param resolutionReceiver: an object that will receive each resolved
            address as it arrives.
        @param hostName: The name of the host to resolve.  If this contains
            non-ASCII code points, they will be converted to IDNA first.
        @param portNumber: The port number that the returned addresses should
            include.
        @param addressTypes: An iterable of implementors of L{IAddress} that
            are acceptable values for C{resolutionReceiver} to receive to its
            L{addressResolved <IResolutionReceiver.addressResolved>}.  In
            practice, this means an iterable containing
            L{twisted.internet.address.IPv4Address},
            L{twisted.internet.address.IPv6Address}, both, or neither.
        @param transportSemantics: A string describing the semantics of the
            transport; either C{'TCP'} for stream-oriented transports or
            C{'UDP'} for datagram-oriented; see
            L{twisted.internet.address.IPv6Address.type} and
            L{twisted.internet.address.IPv4Address.type}.

        @return: The resolution in progress.
        """