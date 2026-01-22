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
class GAIResolver:
    """
    L{IHostnameResolver} implementation that resolves hostnames by calling
    L{getaddrinfo} in a thread.
    """

    def __init__(self, reactor: IReactorThreads, getThreadPool: Optional[Callable[[], 'ThreadPool']]=None, getaddrinfo: Callable[[str, int, int, int], _GETADDRINFO_RESULT]=getaddrinfo):
        """
        Create a L{GAIResolver}.

        @param reactor: the reactor to schedule result-delivery on
        @type reactor: L{IReactorThreads}

        @param getThreadPool: a function to retrieve the thread pool to use for
            scheduling name resolutions.  If not supplied, the use the given
            C{reactor}'s thread pool.
        @type getThreadPool: 0-argument callable returning a
            L{twisted.python.threadpool.ThreadPool}

        @param getaddrinfo: a reference to the L{getaddrinfo} to use - mainly
            parameterized for testing.
        @type getaddrinfo: callable with the same signature as L{getaddrinfo}
        """
        self._reactor = reactor
        self._getThreadPool = reactor.getThreadPool if getThreadPool is None else getThreadPool
        self._getaddrinfo = getaddrinfo

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
        pool = self._getThreadPool()
        addressFamily = _typesToAF[_any if addressTypes is None else frozenset(addressTypes)]
        socketType = _transportToSocket[transportSemantics]

        def get() -> _GETADDRINFO_RESULT:
            try:
                return self._getaddrinfo(hostName, portNumber, addressFamily, socketType)
            except gaierror:
                return []
        d = deferToThreadPool(self._reactor, pool, get)
        resolution = HostResolution(hostName)
        resolutionReceiver.resolutionBegan(resolution)

        @d.addCallback
        def deliverResults(result: _GETADDRINFO_RESULT) -> None:
            for family, socktype, proto, cannoname, sockaddr in result:
                addrType = _afToType[family]
                resolutionReceiver.addressResolved(addrType(_socktypeToType.get(socktype, 'TCP'), *sockaddr))
            resolutionReceiver.resolutionComplete()
        return resolution