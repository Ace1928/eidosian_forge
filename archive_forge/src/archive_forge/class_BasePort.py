import builtins
import socket  # needed only for sync-dns
import warnings
from abc import ABC, abstractmethod
from heapq import heapify, heappop, heappush
from traceback import format_stack
from types import FrameType
from typing import (
from zope.interface import classImplements, implementer
from twisted.internet import abstract, defer, error, fdesc, main, threads
from twisted.internet._resolver import (
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory
from twisted.python import log, reflect
from twisted.python.failure import Failure
from twisted.python.runtime import platform, seconds as runtimeSeconds
from ._signals import SignalHandling, _WithoutSignalHandling, _WithSignalHandling
from twisted.python import threadable
class BasePort(abstract.FileDescriptor):
    """Basic implementation of a ListeningPort.

    Note: This does not actually implement IListeningPort.
    """
    addressFamily: socket.AddressFamily = None
    socketType: socket.SocketKind = None

    def createInternetSocket(self) -> socket.socket:
        s = socket.socket(self.addressFamily, self.socketType)
        s.setblocking(False)
        fdesc._setCloseOnExec(s.fileno())
        return s

    def doWrite(self) -> Optional[Failure]:
        """Raises a RuntimeError"""
        raise RuntimeError('doWrite called on a %s' % reflect.qual(self.__class__))