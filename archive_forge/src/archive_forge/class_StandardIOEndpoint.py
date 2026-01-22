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
@implementer(interfaces.IStreamServerEndpoint)
class StandardIOEndpoint:
    """
    A Standard Input/Output endpoint

    @ivar _stdio: a callable, like L{stdio.StandardIO}, which takes an
        L{IProtocol} provider and a C{reactor} keyword argument (interface
        dependent upon your platform).
    """
    _stdio = StandardIO

    def __init__(self, reactor):
        """
        @param reactor: The reactor for the endpoint.
        """
        self._reactor = reactor

    def listen(self, stdioProtocolFactory):
        """
        Implement L{IStreamServerEndpoint.listen} to listen on stdin/stdout
        """
        return defer.execute(self._stdio, stdioProtocolFactory.buildProtocol(PipeAddress()), reactor=self._reactor)