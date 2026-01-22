from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IFileDescriptorReceiver(Interface):
    """
    Protocols may implement L{IFileDescriptorReceiver} to receive file
    descriptors sent to them.  This is useful in conjunction with
    L{IUNIXTransport}, which allows file descriptors to be sent between
    processes on a single host.
    """

    def fileDescriptorReceived(descriptor: int) -> None:
        """
        Called when a file descriptor is received over the connection.

        @param descriptor: The descriptor which was received.

        @return: L{None}
        """