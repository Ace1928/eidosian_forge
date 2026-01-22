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
@staticmethod
def _hostAsBytesAndText(host):
    """
        For various reasons (documented in the C{@ivar}'s in the class
        docstring) we need both a textual and a binary representation of the
        hostname given to the constructor.  For compatibility and convenience,
        we accept both textual and binary representations of the hostname, save
        the form that was passed, and convert into the other form.  This is
        mostly just because L{HostnameAddress} chose somewhat poorly to define
        its attribute as bytes; hopefully we can find a compatible way to clean
        this up in the future and just operate in terms of text internally.

        @param host: A hostname to convert.
        @type host: L{bytes} or C{str}

        @return: a 3-tuple of C{(invalid, bytes, text)} where C{invalid} is a
            boolean indicating the validity of the hostname, C{bytes} is a
            binary representation of C{host}, and C{text} is a textual
            representation of C{host}.
        """
    if isinstance(host, bytes):
        if isIPAddress(host) or isIPv6Address(host):
            return (False, host, host.decode('ascii'))
        else:
            try:
                return (False, host, _idnaText(host))
            except UnicodeError:
                host = host.decode('charmap')
    else:
        host = normalize('NFC', host)
        if isIPAddress(host) or isIPv6Address(host):
            return (False, host.encode('ascii'), host)
        else:
            try:
                return (False, _idnaBytes(host), host)
            except UnicodeError:
                pass
    asciibytes = host.encode('ascii', 'backslashreplace')
    return (True, asciibytes, asciibytes.decode('ascii'))