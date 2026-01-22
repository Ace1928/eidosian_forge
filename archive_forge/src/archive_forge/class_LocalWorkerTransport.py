import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, TextIO, TypeVar
from unittest import TestCase
from zope.interface import implementer
from attrs import frozen
from typing_extensions import Protocol, TypedDict
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.internet.protocol import ProcessProtocol
from twisted.logger import Logger
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedObject
from twisted.trial._dist import (
from twisted.trial._dist.workerreporter import WorkerReporter
from twisted.trial.reporter import TestResult
from twisted.trial.runner import TestLoader, TrialSuite
from twisted.trial.unittest import Todo
from .stream import StreamOpen, StreamReceiver, StreamWrite
@implementer(ITransport)
class LocalWorkerTransport:
    """
    A stub transport implementation used to support L{AMP} over a
    L{ProcessProtocol} transport.
    """

    def __init__(self, transport):
        self._transport = transport

    def write(self, data):
        """
        Forward data to transport.
        """
        self._transport.writeToChild(_WORKER_AMP_STDIN, data)

    def writeSequence(self, sequence):
        """
        Emulate C{writeSequence} by iterating data in the C{sequence}.
        """
        for data in sequence:
            self._transport.writeToChild(_WORKER_AMP_STDIN, data)

    def loseConnection(self):
        """
        Closes the transport.
        """
        self._transport.loseConnection()

    def getHost(self):
        """
        Return a L{LocalWorkerAddress} instance.
        """
        return LocalWorkerAddress()

    def getPeer(self):
        """
        Return a L{LocalWorkerAddress} instance.
        """
        return LocalWorkerAddress()