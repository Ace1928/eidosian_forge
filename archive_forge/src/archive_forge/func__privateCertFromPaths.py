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
def _privateCertFromPaths(certificatePath, keyPath):
    """
    Parse a certificate path and key path, either or both of which might be
    L{None}, into a certificate object.

    @param certificatePath: the certificate path
    @type certificatePath: L{bytes} or L{unicode} or L{None}

    @param keyPath: the private key path
    @type keyPath: L{bytes} or L{unicode} or L{None}

    @return: a L{PrivateCertificate} or L{None}
    """
    if certificatePath is None:
        return None
    certBytes = FilePath(certificatePath).getContent()
    if keyPath is None:
        return PrivateCertificate.loadPEM(certBytes)
    else:
        return PrivateCertificate.fromCertificateAndKeyPair(Certificate.loadPEM(certBytes), KeyPair.load(FilePath(keyPath).getContent(), 1))