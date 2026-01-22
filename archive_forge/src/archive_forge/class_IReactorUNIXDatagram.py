from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorUNIXDatagram(Interface):
    """
    Datagram UNIX socket methods.
    """

    def connectUNIXDatagram(address: str, protocol: 'ConnectedDatagramProtocol', maxPacketSize: int, mode: int, bindAddress: Optional[Tuple[str, int]]) -> IConnector:
        """
        Connect a client protocol to a datagram UNIX socket.

        @param address: a path to a unix socket on the filesystem.
        @param protocol: a L{twisted.internet.protocol.ConnectedDatagramProtocol} instance
        @param maxPacketSize: maximum packet size to accept
        @param mode: The mode (B{not} umask) to set on the unix socket.  See
            platform specific documentation for information about how this
            might affect connection attempts.

        @param bindAddress: address to bind to

        @return: An object which provides L{IConnector}.
        """

    def listenUNIXDatagram(address: str, protocol: 'DatagramProtocol', maxPacketSize: int, mode: int) -> 'IListeningPort':
        """
        Listen on a datagram UNIX socket.

        @param address: a path to a unix socket on the filesystem.
        @param protocol: a L{twisted.internet.protocol.DatagramProtocol} instance.
        @param maxPacketSize: maximum packet size to accept
        @param mode: The mode (B{not} umask) to set on the unix socket.  See
            platform specific documentation for information about how this
            might affect connection attempts.

        @return: An object which provides L{IListeningPort}.
        """