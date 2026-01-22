import os
import socket
import traceback
from unittest import skipIf
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor
from twisted.internet.tcp import EINPROGRESS, EWOULDBLOCK
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
@implementer(IReadDescriptor)
class RemovingDescriptor:
    """
    A read descriptor which removes itself from the reactor as soon as it
    gets a chance to do a read and keeps track of when its own C{fileno}
    method is called.

    @ivar insideReactor: A flag which is true as long as the reactor has
        this descriptor as a reader.

    @ivar calls: A list of the bottom of the call stack for any call to
        C{fileno} when C{insideReactor} is false.
    """

    def __init__(self, reactor):
        self.reactor = reactor
        self.insideReactor = False
        self.calls = []
        self.read, self.write = socketpair()

    def start(self):
        self.insideReactor = True
        self.reactor.addReader(self)
        self.write.send(b'a')

    def logPrefix(self):
        return 'foo'

    def doRead(self):
        self.reactor.removeReader(self)
        self.insideReactor = False
        self.reactor.stop()
        self.read.close()
        self.write.close()

    def fileno(self):
        if not self.insideReactor:
            self.calls.append(traceback.extract_stack(limit=5)[:-1])
        return self.read.fileno()

    def connectionLost(self, reason):
        pass