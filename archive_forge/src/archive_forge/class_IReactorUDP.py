from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorUDP(Interface):
    """
    UDP socket methods.
    """

    def listenUDP(port: int, protocol: 'DatagramProtocol', interface: str, maxPacketSize: int) -> 'IListeningPort':
        """
        Connects a given L{DatagramProtocol} to the given numeric UDP port.

        @param port: A port number on which to listen.
        @param protocol: A L{DatagramProtocol} instance which will be
            connected to the given C{port}.
        @param interface: The local IPv4 or IPv6 address to which to bind;
            defaults to '', ie all IPv4 addresses.
        @param maxPacketSize: The maximum packet size to accept.

        @return: object which provides L{IListeningPort}.
        """