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
def _parseClientTCP(*args, **kwargs):
    """
    Perform any argument value coercion necessary for TCP client parameters.

    Valid positional arguments to this function are host and port.

    Valid keyword arguments to this function are all L{IReactorTCP.connectTCP}
    arguments.

    @return: The coerced values as a C{dict}.
    """
    if len(args) == 2:
        kwargs['port'] = int(args[1])
        kwargs['host'] = args[0]
    elif len(args) == 1:
        if 'host' in kwargs:
            kwargs['port'] = int(args[0])
        else:
            kwargs['host'] = args[0]
    try:
        kwargs['port'] = int(kwargs['port'])
    except KeyError:
        pass
    try:
        kwargs['timeout'] = int(kwargs['timeout'])
    except KeyError:
        pass
    try:
        kwargs['bindAddress'] = (kwargs['bindAddress'], 0)
    except KeyError:
        pass
    return kwargs