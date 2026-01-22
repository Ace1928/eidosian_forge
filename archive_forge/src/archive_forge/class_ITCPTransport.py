from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class ITCPTransport(ITransport):
    """
    A TCP based transport.
    """

    def loseWriteConnection() -> None:
        """
        Half-close the write side of a TCP connection.

        If the protocol instance this is attached to provides
        IHalfCloseableProtocol, it will get notified when the operation is
        done. When closing write connection, as with loseConnection this will
        only happen when buffer has emptied and there is no registered
        producer.
        """

    def abortConnection() -> None:
        """
        Close the connection abruptly.

        Discards any buffered data, stops any registered producer,
        and, if possible, notifies the other end of the unclean
        closure.

        @since: 11.1
        """

    def getTcpNoDelay() -> bool:
        """
        Return if C{TCP_NODELAY} is enabled.
        """

    def setTcpNoDelay(enabled: bool) -> None:
        """
        Enable/disable C{TCP_NODELAY}.

        Enabling C{TCP_NODELAY} turns off Nagle's algorithm. Small packets are
        sent sooner, possibly at the expense of overall throughput.
        """

    def getTcpKeepAlive() -> bool:
        """
        Return if C{SO_KEEPALIVE} is enabled.
        """

    def setTcpKeepAlive(enabled: bool) -> None:
        """
        Enable/disable C{SO_KEEPALIVE}.

        Enabling C{SO_KEEPALIVE} sends packets periodically when the connection
        is otherwise idle, usually once every two hours. They are intended
        to allow detection of lost peers in a non-infinite amount of time.
        """

    def getHost() -> Union['IPv4Address', 'IPv6Address']:
        """
        Returns L{IPv4Address} or L{IPv6Address}.
        """

    def getPeer() -> Union['IPv4Address', 'IPv6Address']:
        """
        Returns L{IPv4Address} or L{IPv6Address}.
        """