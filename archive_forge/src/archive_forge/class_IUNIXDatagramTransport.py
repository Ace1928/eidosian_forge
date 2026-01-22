from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IUNIXDatagramTransport(Interface):
    """
    Transport for UDP PacketProtocols.
    """

    def write(packet: bytes, addr: str) -> None:
        """
        Write packet to given address.
        """

    def getHost() -> 'UNIXAddress':
        """
        Returns L{UNIXAddress}.
        """