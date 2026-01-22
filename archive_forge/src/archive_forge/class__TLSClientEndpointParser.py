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
@implementer(IPlugin, IStreamClientEndpointStringParserWithReactor)
class _TLSClientEndpointParser:
    """
    Stream client endpoint string parser for L{wrapClientTLS} with
    L{HostnameEndpoint}.

    @ivar prefix: See
        L{IStreamClientEndpointStringParserWithReactor.prefix}.
    """
    prefix = 'tls'

    @staticmethod
    def parseStreamClient(reactor, *args, **kwargs):
        """
        Redirects to another function L{_parseClientTLS}; tricks zope.interface
        into believing the interface is correctly implemented, since the
        signature is (C{reactor}, C{*args}, C{**kwargs}).  See
        L{_parseClientTLS} for the specific signature description for this
        endpoint parser.

        @param reactor: The reactor passed to L{clientFromString}.

        @param args: The positional arguments in the endpoint description.
        @type args: L{tuple}

        @param kwargs: The named arguments in the endpoint description.
        @type kwargs: L{dict}

        @return: a client TLS endpoint
        @rtype: L{IStreamClientEndpoint}
        """
        return _parseClientTLS(reactor, *args, **kwargs)