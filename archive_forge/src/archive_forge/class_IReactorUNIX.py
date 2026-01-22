from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorUNIX(Interface):
    """
    UNIX socket methods.
    """

    def connectUNIX(address: str, factory: 'ClientFactory', timeout: float, checkPID: bool) -> IConnector:
        """
        Connect a client protocol to a UNIX socket.

        @param address: a path to a unix socket on the filesystem.
        @param factory: a L{twisted.internet.protocol.ClientFactory} instance
        @param timeout: number of seconds to wait before assuming the connection
            has failed.
        @param checkPID: if True, check for a pid file to verify that a server
            is listening.  If C{address} is a Linux abstract namespace path,
            this must be C{False}.

        @return: An object which provides L{IConnector}.
        """

    def listenUNIX(address: str, factory: 'Factory', backlog: int, mode: int, wantPID: bool) -> 'IListeningPort':
        """
        Listen on a UNIX socket.

        @param address: a path to a unix socket on the filesystem.
        @param factory: a L{twisted.internet.protocol.Factory} instance.
        @param backlog: number of connections to allow in backlog.
        @param mode: The mode (B{not} umask) to set on the unix socket.  See
            platform specific documentation for information about how this
            might affect connection attempts.
        @param wantPID: if True, create a pidfile for the socket.  If C{address}
            is a Linux abstract namespace path, this must be C{False}.

        @return: An object which provides L{IListeningPort}.
        """