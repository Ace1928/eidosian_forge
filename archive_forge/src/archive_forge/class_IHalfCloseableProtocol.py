from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IHalfCloseableProtocol(Interface):
    """
    Implemented to indicate they want notification of half-closes.

    TCP supports the notion of half-closing the connection, e.g.
    closing the write side but still not stopping reading. A protocol
    that implements this interface will be notified of such events,
    instead of having connectionLost called.
    """

    def readConnectionLost() -> None:
        """
        Notification of the read connection being closed.

        This indicates peer did half-close of write side. It is now
        the responsibility of the this protocol to call
        loseConnection().  In addition, the protocol MUST make sure a
        reference to it still exists (i.e. by doing a callLater with
        one of its methods, etc.)  as the reactor will only have a
        reference to it if it is writing.

        If the protocol does not do so, it might get garbage collected
        without the connectionLost method ever being called.
        """

    def writeConnectionLost() -> None:
        """
        Notification of the write connection being closed.

        This will never be called for TCP connections as TCP does not
        support notification of this type of half-close.
        """