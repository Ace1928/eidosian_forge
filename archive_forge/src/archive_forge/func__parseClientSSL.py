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
def _parseClientSSL(*args, **kwargs):
    """
    Perform any argument value coercion necessary for SSL client parameters.

    Valid keyword arguments to this function are all L{IReactorSSL.connectSSL}
    arguments except for C{contextFactory}.  Instead, C{certKey} (the path name
    of the certificate file) C{privateKey} (the path name of the private key
    associated with the certificate) are accepted and used to construct a
    context factory.

    Valid positional arguments to this function are host and port.

    @keyword caCertsDir: The one parameter which is not part of
        L{IReactorSSL.connectSSL}'s signature, this is a path name used to
        construct a list of certificate authority certificates.  The directory
        will be scanned for files ending in C{.pem}, all of which will be
        considered valid certificate authorities for this connection.
    @type caCertsDir: L{str}

    @keyword hostname: The hostname to use for validating the server's
        certificate.
    @type hostname: L{unicode}

    @return: The coerced values as a L{dict}.
    """
    kwargs = _parseClientTCP(*args, **kwargs)
    return _parseClientSSLOptions(kwargs)