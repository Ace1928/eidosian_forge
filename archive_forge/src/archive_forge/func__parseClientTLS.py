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
def _parseClientTLS(reactor, host, port, timeout=b'30', bindAddress=None, certificate=None, privateKey=None, trustRoots=None, endpoint=None, **kwargs):
    """
    Internal method to construct an endpoint from string parameters.

    @param reactor: The reactor passed to L{clientFromString}.

    @param host: The hostname to connect to.
    @type host: L{bytes} or L{unicode}

    @param port: The port to connect to.
    @type port: L{bytes} or L{unicode}

    @param timeout: For each individual connection attempt, the number of
        seconds to wait before assuming the connection has failed.
    @type timeout: L{bytes} or L{unicode}

    @param bindAddress: The address to which to bind outgoing connections.
    @type bindAddress: L{bytes} or L{unicode}

    @param certificate: a string representing a filesystem path to a
        PEM-encoded certificate.
    @type certificate: L{bytes} or L{unicode}

    @param privateKey: a string representing a filesystem path to a PEM-encoded
        certificate.
    @type privateKey: L{bytes} or L{unicode}

    @param endpoint: an optional string endpoint description of an endpoint to
        wrap; if this is passed then C{host} is used only for certificate
        verification.
    @type endpoint: L{bytes} or L{unicode}

    @return: a client TLS endpoint
    @rtype: L{IStreamClientEndpoint}
    """
    if kwargs:
        raise TypeError('unrecognized keyword arguments present', list(kwargs.keys()))
    host = host if isinstance(host, str) else host.decode('utf-8')
    bindAddress = bindAddress if isinstance(bindAddress, str) or bindAddress is None else bindAddress.decode('utf-8')
    port = int(port)
    timeout = int(timeout)
    return wrapClientTLS(optionsForClientTLS(host, trustRoot=_parseTrustRootPath(trustRoots), clientCertificate=_privateCertFromPaths(certificate, privateKey)), clientFromString(reactor, endpoint) if endpoint is not None else HostnameEndpoint(reactor, _idnaBytes(host), port, timeout, bindAddress))