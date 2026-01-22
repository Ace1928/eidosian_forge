import os
import re
import socket
import warnings
from typing import Optional, Sequence, Type
from unicodedata import normalize
from zope.interface import directlyProvides, implementer, provider
from constantly import NamedConstant, Names
from incremental import Version
from twisted.internet import defer, error, fdesc, interfaces, threads
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.address import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, ProcessProtocol, Protocol
from twisted.internet._resolver import HostResolution
from twisted.internet.defer import Deferred
from twisted.internet.task import LoopingCall
from twisted.logger import Logger
from twisted.plugin import IPlugin, getPlugins
from twisted.python import deprecate, log
from twisted.python.compat import _matchingString, iterbytes, nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.systemd import ListenFDs
from ._idna import _idnaBytes, _idnaText
@implementer(IPlugin, IStreamServerEndpointStringParser)
class _SystemdParser:
    """
    Stream server endpoint string parser for the I{systemd} endpoint type.

    @ivar prefix: See L{IStreamServerEndpointStringParser.prefix}.

    @ivar _sddaemon: A L{ListenFDs} instance used to translate an index into an
        actual file descriptor.
    """
    _sddaemon = ListenFDs.fromEnvironment()
    prefix = 'systemd'

    def _parseServer(self, reactor: IReactorSocket, domain: str, index: Optional[str]=None, name: Optional[str]=None) -> AdoptedStreamServerEndpoint:
        """
        Internal parser function for L{_parseServer} to convert the string
        arguments for a systemd server endpoint into structured arguments for
        L{AdoptedStreamServerEndpoint}.

        @param reactor: An L{IReactorSocket} provider.

        @param domain: The domain (or address family) of the socket inherited
            from systemd.  This is a string like C{"INET"} or C{"UNIX"}, ie
            the name of an address family from the L{socket} module, without
            the C{"AF_"} prefix.

        @param index: If given, the decimal representation of an integer
            giving the offset into the list of file descriptors inherited from
            systemd.  Since the order of descriptors received from systemd is
            hard to predict, this option should only be used if only one
            descriptor is being inherited.  Even in that case, C{name} is
            probably a better idea.  Either C{index} or C{name} must be given.

        @param name: If given, the name (as defined by C{FileDescriptorName}
            in the C{[Socket]} section of a systemd service definition) of an
            inherited file descriptor.  Either C{index} or C{name} must be
            given.

        @return: An L{AdoptedStreamServerEndpoint} which will adopt the
            inherited listening port when it is used to listen.
        """
        if (index is None) == (name is None):
            raise ValueError('Specify exactly one of descriptor index or name')
        if index is not None:
            fileno = self._sddaemon.inheritedDescriptors()[int(index)]
        else:
            assert name is not None
            fileno = self._sddaemon.inheritedNamedDescriptors()[name]
        addressFamily = getattr(socket, 'AF_' + domain)
        return AdoptedStreamServerEndpoint(reactor, fileno, addressFamily)

    def parseStreamServer(self, reactor, *args, **kwargs):
        return self._parseServer(reactor, *args, **kwargs)